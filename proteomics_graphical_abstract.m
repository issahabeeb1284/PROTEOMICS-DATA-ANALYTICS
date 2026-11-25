%% PRECISION PROTEOMICS VISUALIZATION PIPELINE (ANONYMIZED) - MATLAB
% =========================================================================
% Builds a publication-grade, multi-panel graphical abstract from a
% label-free quantitative proteomics experiment with several genotypes and
% biological replicates. Every genotype is shown only as "Sample A",
% "Sample B", "Sample C", ...: Sample A is the reference (control) and every
% other sample is compared against it. Protein and pathway identifiers are
% replaced with neutral codes (Protein_0001, Pathway 1, ...) so no gene or
% sample name is shown.
%
% Panels
%   a  PCA of all replicates
%   b  Volcano plot for each sample vs Sample A (shared axes)
%   c  Up/down differential-abundance counts (diverging bars)
%   d  UpSet-style overlap of the differentially abundant protein sets
%   e  Clustered z-score heatmap of the top differentially abundant proteins
%   f  Pathway over-representation dot plot (hypergeometric test, BH-FDR)
%
% Analysis steps
%   log2 (if needed) -> valid-value filter -> median normalization ->
%   left-shifted normal imputation (MNAR) -> Welch t-test -> BH FDR
%
% Usage
%   Leave INPUT_FILE empty to run on simulated demo data, or point it to a
%   CSV whose first column holds protein IDs and whose other columns are
%   replicates named "<group>_R<n>" (e.g. WT_R1, KO1_R1, ...). Groups are
%   relabeled Sample A, B, C, ... in column order (reference first).
%   PATHWAY_FILE (optional): CSV with two columns, Pathway and Protein.
%
% Requirements: base MATLAB R2020a+ (exportgraphics). The Statistics and
% Machine Learning Toolbox is used for heatmap clustering only if present.
% =========================================================================
clear; close all; clc;

%% ---------------- USER SETTINGS ----------------------------------------
INPUT_FILE         = '';      % '' -> simulated demo data
PATHWAY_FILE       = '';      % required when INPUT_FILE is set
REFERENCE_GROUP    = '';      % '' -> first column group is Sample A
KEEP_PATHWAY_NAMES = false;   % false -> Pathway 1, Pathway 2, ...
OUT_DIR            = fullfile(fileparts(mfilename('fullpath')), 'output_matlab');

FDR_THRESH         = 0.05;
LOG2FC_THRESH      = 1.0;
MIN_VALID_FRACTION = 0.70;
HEATMAP_TOP_N      = 60;
SEED               = 42;

%% ---------------- THEME (validated palette, fixed order) ---------------
C.ink   = hex2rgb('#0b0b0b');  C.ink2  = hex2rgb('#52514e');
C.muted = hex2rgb('#898781');  C.grid  = hex2rgb('#e1e0d9');
C.ns    = hex2rgb('#c3c2b7');  C.up    = hex2rgb('#e34948');
C.down  = hex2rgb('#2a78d6');  C.ref   = hex2rgb('#52514e');
C.hues  = cell2mat(cellfun(@hex2rgb, {'#2a78d6','#1baf7a','#4a3aa7', ...
            '#eda100','#e87ba4','#008300','#eb6834'}, 'UniformOutput', false)');
C.div   = makeCmap({'#184f95','#2a78d6','#f0efec','#e34948','#8a2323'});
C.seq   = makeCmap({'#cde2fb','#6da7ec','#2a78d6','#184f95'});
FONT    = 'Arial';

%% ---------------- DATA ---------------------------------------------------
rng(SEED);
if isempty(INPUT_FILE)
    fprintf('No INPUT_FILE given: running on simulated demo data.\n');
    [X, proteins, colGroup, samples, pw] = simulateDataset(3000, 4, 4);
else
    [X, proteins, colGroup, samples, pw] = loadDataset(INPUT_FILE, PATHWAY_FILE, ...
        REFERENCE_GROUP, KEEP_PATHWAY_NAMES);
end
nS = numel(samples);
sCol = zeros(nS, 3);  sCol(1,:) = C.ref;
for i = 2:nS, sCol(i,:) = C.hues(mod(i-2, size(C.hues,1)) + 1, :); end

%% ---------------- PREPROCESSING ------------------------------------------
if max(X(:), [], 'omitnan') > 100
    X(X <= 0) = NaN;  X = log2(X);
end
validFrac = zeros(size(X,1), nS);
for s = 1:nS
    validFrac(:,s) = mean(~isnan(X(:, colGroup == s)), 2);
end
keep = any(validFrac >= MIN_VALID_FRACTION, 2);
fprintf('Proteins after valid-value filter: %d / %d\n', sum(keep), numel(keep));
X = X(keep,:);  proteins = proteins(keep);
med = median(X, 1, 'omitnan');
X = X - med + mean(med);                          % median normalization
for j = 1:size(X,2)                               % left-shifted imputation
    miss = isnan(X(:,j));
    if any(miss)
        mu = mean(X(:,j), 'omitnan');  sd = std(X(:,j), 'omitnan');
        X(miss,j) = (mu - 1.8*sd) + 0.3*sd*randn(sum(miss),1);
    end
end

%% ---------------- DIFFERENTIAL ABUNDANCE (Welch t-test + BH) -------------
nP = size(X,1);
L2FC = zeros(nP, nS-1);  P = L2FC;  Q = L2FC;  CALL = zeros(nP, nS-1);  % 1 up, -1 down
ref = X(:, colGroup == 1);
for s = 2:nS
    tst = X(:, colGroup == s);
    [P(:,s-1), L2FC(:,s-1)] = welchT(tst, ref);
    Q(:,s-1) = bhFDR(P(:,s-1));
    sig = Q(:,s-1) < FDR_THRESH;
    CALL(sig & L2FC(:,s-1) >  LOG2FC_THRESH, s-1) =  1;
    CALL(sig & L2FC(:,s-1) < -LOG2FC_THRESH, s-1) = -1;
end
for s = 2:nS
    fprintf('%s: %d up, %d down\n', samples{s}, sum(CALL(:,s-1)==1), sum(CALL(:,s-1)==-1));
end

%% ---------------- PATHWAY OVER-REPRESENTATION ----------------------------
pwNames = unique(pw.Pathway, 'stable');
memb = false(nP, numel(pwNames));
for k = 1:numel(pwNames)
    memb(:,k) = ismember(proteins, pw.Protein(strcmp(pw.Pathway, pwNames{k})));
end
okSet = sum(memb,1) >= 10;  memb = memb(:,okSet);  pwNames = pwNames(okSet);
nPW = numel(pwNames);
ovl = zeros(nPW, nS-1);  pEnr = ones(nPW, nS-1);  qEnr = pEnr;
for s = 1:nS-1
    hits = CALL(:,s) ~= 0;  n = sum(hits);
    for k = 1:nPW
        K = sum(memb(:,k));  kk = sum(hits & memb(:,k));  ovl(k,s) = kk;
        if kk > 0, pEnr(k,s) = hygeUpper(kk, nP, K, n); end
    end
    qEnr(:,s) = bhFDR(pEnr(:,s));
end

%% ---------------- SAVE TABLES --------------------------------------------
if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end
rows = {};
callTxt = {'down','ns','up'};
for s = 2:nS
    rows = [rows; [proteins, repmat(samples(s), nP, 1), num2cell(L2FC(:,s-1)), ...
        num2cell(P(:,s-1)), num2cell(Q(:,s-1)), callTxt(CALL(:,s-1)+2)']]; %#ok<AGROW>
end
writetable(cell2table(rows, 'VariableNames', ...
    {'Protein','Sample','log2FC','pvalue','adj_pvalue','call'}), ...
    fullfile(OUT_DIR, 'differential_abundance.csv'));

%% =================== FIGURE =============================================
fig = figure('Units','centimeters','Position',[2 2 18.3 16.5],'Color','w', ...
             'PaperPositionMode','auto','Visible','on');
set(fig, 'DefaultAxesFontName', FONT, 'DefaultTextFontName', FONT, ...
    'DefaultAxesFontSize', 6.5, 'DefaultAxesXColor', C.ink2, ...
    'DefaultAxesYColor', C.ink2, 'DefaultAxesLineWidth', 0.6, ...
    'DefaultAxesTickDir', 'out', 'DefaultAxesBox', 'off', ...
    'DefaultAxesTickLength', [0.015 0.015]);

annotation(fig, 'textbox', [0 0.96 1 0.04], 'String', ...
    'Precision proteomics: genotype-dependent remodeling relative to Sample A', ...
    'EdgeColor','none','HorizontalAlignment','center','FontWeight','bold', ...
    'FontSize', 9.5, 'Color', C.ink, 'FontName', FONT);

% ---- (a) PCA --------------------------------------------------------------
ax = axes(fig, 'Position', [0.075 0.70 0.17 0.21]);  hold(ax,'on');
Xc = X' - mean(X', 1);
[U, S, ~] = svd(Xc, 'econ');  pcs = U*S;  ev = diag(S).^2;  ev = 100*ev/sum(ev);
for s = 1:nS
    m = colGroup == s;
    scatter(ax, pcs(m,1), pcs(m,2), 30, sCol(s,:), 'filled', ...
        'MarkerEdgeColor','w','LineWidth',0.6);
    yOff = 0.07 * (max(pcs(:,2)) - min(pcs(:,2)));
    text(ax, mean(pcs(m,1)), max(pcs(m,2)) + yOff, strrep(samples{s},'Sample ',''), ...
        'VerticalAlignment','bottom','HorizontalAlignment','center', ...
        'FontWeight','bold','FontSize',7,'Color',C.ink);
end
styleAxes(ax, C);  grid(ax,'on');  padLimits(ax, 0.18);
xlabel(ax, sprintf('PC1 (%.1f%%)', ev(1)), 'Color', C.ink, 'FontSize', 7);
ylabel(ax, sprintf('PC2 (%.1f%%)', ev(2)), 'Color', C.ink, 'FontSize', 7);
title(ax, 'Replicate structure (PCA)', 'FontWeight','normal','FontSize',7.5,'Color',C.ink);
panelLetter(fig, ax, 'a');

% ---- (b) Volcanos ---------------------------------------------------------
nlq = -log10(max(Q, 1e-300));
xl  = ceil(quantileSimple(abs(L2FC(:)), 0.999) + 0.5);
yl  = quantileSimple(nlq(:), 0.999) * 1.12;
vx0 = 0.33;  vw = (0.97 - vx0 - 0.02*(nS-2)) / (nS-1);
for s = 1:nS-1
    ax = axes(fig, 'Position', [vx0 + (s-1)*(vw+0.02) 0.70 vw 0.21]);  hold(ax,'on');
    groups = {CALL(:,s)==0, C.ns, 5, 0.55; CALL(:,s)==-1, C.down, 7, 0.9; CALL(:,s)==1, C.up, 7, 0.9};
    for g = 1:3
        m = groups{g,1};
        scatter(ax, L2FC(m,s), nlq(m,s), groups{g,3}, groups{g,2}, 'filled', ...
            'MarkerFaceAlpha', groups{g,4}, 'MarkerEdgeColor','none');
    end
    xline(ax, [-LOG2FC_THRESH LOG2FC_THRESH], '--', 'Color', C.muted, 'LineWidth', 0.5);
    yline(ax, -log10(FDR_THRESH), '--', 'Color', C.muted, 'LineWidth', 0.5);
    xlim(ax, [-xl xl]);  ylim(ax, [0 yl]);  styleAxes(ax, C);
    text(ax, 0.04, 0.97, sprintf('\\nabla %d', sum(CALL(:,s)==-1)), 'Units','normalized', ...
        'VerticalAlignment','top','Color',C.ink2,'FontSize',6.5);
    text(ax, 0.96, 0.97, sprintf('%d \\Delta', sum(CALL(:,s)==1)), 'Units','normalized', ...
        'VerticalAlignment','top','HorizontalAlignment','right','Color',C.ink2,'FontSize',6.5);
    title(ax, sprintf('%s vs Sample A', samples{s+1}), 'FontWeight','normal', ...
        'FontSize',7.5,'Color',C.ink);
    xlabel(ax, 'log_2FC', 'Color', C.ink, 'FontSize', 7);
    if s == 1
        ylabel(ax, '-log_{10}(FDR)', 'Color', C.ink, 'FontSize', 7);
        panelLetter(fig, ax, 'b');
    else
        ax.YTickLabel = [];
    end
end

% ---- (c) Up/down counts ---------------------------------------------------
ax = axes(fig, 'Position', [0.10 0.41 0.16 0.19]);  hold(ax,'on');
nUp = sum(CALL==1,1);  nDn = sum(CALL==-1,1);  y = (nS-1):-1:1;
hU = barh(ax, y,  nUp, 0.62, 'FaceColor', C.up,   'EdgeColor','none');
hD = barh(ax, y, -nDn, 0.62, 'FaceColor', C.down, 'EdgeColor','none');
mx = max([nUp nDn]);
for i = 1:nS-1
    text(ax,  nUp(i) + 0.04*mx, y(i), num2str(nUp(i)), 'FontSize',6.5,'Color',C.ink2);
    text(ax, -nDn(i) - 0.04*mx, y(i), num2str(nDn(i)), 'FontSize',6.5,'Color',C.ink2, ...
        'HorizontalAlignment','right');
end
xline(ax, 0, 'Color', C.ink2, 'LineWidth', 0.6);
xlim(ax, [-1.6 1.6]*mx);  ylim(ax, [0.4 nS-0.6]);
set(ax, 'YTick', fliplr(y), 'YTickLabel', samples(end:-1:2), 'TickLength',[0 0]);
ax.YAxis.Visible = 'on';  ax.YAxis.Color = 'none';  ax.YAxis.TickLabelColor = C.ink2;
xt = ax.XTick;  ax.XTickLabel = arrayfun(@(v) num2str(abs(v)), xt, 'UniformOutput', false);
styleAxes(ax, C);  ax.YColor = 'none';
for i = 1:nS-1   % y labels drawn as text because the y axis is hidden
    text(ax, -1.6*mx, y(i), [samples{i+1} '  '], 'HorizontalAlignment','right', ...
        'FontSize',6.5,'Color',C.ink2);
end
xlabel(ax, 'Differentially abundant proteins', 'Color', C.ink, 'FontSize', 7);
title(ax, 'Direction of change', 'FontWeight','normal','FontSize',7.5,'Color',C.ink);
lg = legend(ax, [hU hD], {'Up vs A','Down vs A'}, 'Orientation','horizontal', ...
    'Box','off','FontSize',6.5,'Position',[0.055 0.335 0.24 0.025]);
lg.ItemTokenSize = [8 8];
panelLetter(fig, ax, 'c');

% ---- (d) UpSet-style overlap ----------------------------------------------
k = nS - 1;  sigM = CALL ~= 0;  combos = [];  counts = [];
for mask = 1:(2^k - 1)
    inSet = bitget(mask, 1:k) == 1;
    excl  = all(sigM(:, inSet), 2) & ~any(sigM(:, ~inSet), 2);
    if sum(excl) > 0, combos = [combos; inSet]; counts = [counts; sum(excl)]; end %#ok<AGROW>
end
[counts, ord] = sort(counts, 'descend');  combos = combos(ord,:);
nc = min(10, numel(counts));  counts = counts(1:nc);  combos = combos(1:nc,:);
axB = axes(fig, 'Position', [0.40 0.475 0.21 0.125]);  hold(axB,'on');
bar(axB, 1:nc, counts, 0.62, 'FaceColor', C.ink2, 'EdgeColor','none');
for i = 1:nc
    text(axB, i, counts(i), num2str(counts(i)), 'HorizontalAlignment','center', ...
        'VerticalAlignment','bottom','FontSize',6,'Color',C.ink2);
end
xlim(axB, [0.4 nc+0.6]);  ylim(axB, [0 max(counts)*1.18]);  styleAxes(axB, C);
axB.XColor = 'none';
ylabel(axB, 'Proteins', 'Color', C.ink, 'FontSize', 7);
title(axB, sprintf('Shared vs unique changes (n = %d)', sum(any(sigM,2))), ...
    'FontWeight','normal','FontSize',7.5,'Color',C.ink);
panelLetter(fig, axB, 'd');
axM = axes(fig, 'Position', [0.40 0.38 0.21 0.075]);  hold(axM,'on');
for i = 1:nc
    yy = find(combos(i,:));  yy = k + 1 - yy;
    if numel(yy) > 1, plot(axM, [i i], [min(yy) max(yy)], '-', 'Color', C.ink2, 'LineWidth', 1); end
    for s = 1:k
        on = combos(i,s);
        if on, col = sCol(s+1,:); sz = 22; else, col = C.grid; sz = 14; end
        scatter(axM, i, k+1-s, sz, col, 'filled', 'MarkerEdgeColor','w','LineWidth',0.5);
    end
end
xlim(axM, [0.4 nc+0.6]);  ylim(axM, [0.4 k+0.6]);
set(axM, 'YTick', 1:k, 'YTickLabel', samples(end:-1:2), 'TickLength',[0 0]);
axM.XColor = 'none';  axM.YAxis.Color = 'none';  axM.YAxis.TickLabelColor = C.ink2;
axM.FontName = FONT;  axM.FontSize = 6.5;

% ---- (e) Clustered z-score heatmap -----------------------------------------
per = max(floor(HEATMAP_TOP_N / (nS-1)), 1);  top = [];
for s = 1:nS-1
    idx = find(CALL(:,s) ~= 0);
    [~, o] = sort(Q(idx,s));  idx = idx(o);
    top = [top; idx(1:min(per, numel(idx)))]; %#ok<AGROW>
end
top = unique(top, 'stable');  top = top(1:min(HEATMAP_TOP_N, numel(top)));
[~, colOrd] = sort(colGroup);  colOrd = colOrd(:)';
Z = X(top, colOrd);  Z = (Z - mean(Z,2)) ./ std(Z, 0, 2);
rowOrd = clusterRows(Z);  Z = Z(rowOrd,:);
nCol = size(Z,2);
axS = axes(fig, 'Position', [0.67 0.577 0.225 0.015]);  hold(axS,'on');
gC = colGroup(colOrd);  bounds = [0 cumsum(arrayfun(@(s) sum(gC==s), 1:nS))];
for s = 1:nS
    rectangle(axS, 'Position', [bounds(s)+0.6, 0, bounds(s+1)-bounds(s)-0.2, 1], ...
        'FaceColor', sCol(s,:), 'EdgeColor', 'none');
end
xlim(axS, [0.5 nCol+0.5]);  ylim(axS, [0 1]);  axis(axS, 'off');
title(axS, 'Proteome signature (row z-score)', 'FontWeight','normal','FontSize',7.5, ...
    'Color',C.ink, 'Units','normalized', 'Position',[0.5 2.0 0]);
axH = axes(fig, 'Position', [0.67 0.07 0.225 0.50]);
imagesc(axH, Z, [-2.5 2.5]);  colormap(axH, C.div);  hold(axH,'on');
for b = bounds(2:end-1), xline(axH, b+0.5, 'Color','w','LineWidth',1.5); end
centers = (bounds(1:end-1) + bounds(2:end)) / 2 + 0.5;
set(axH, 'XTick', centers, 'XTickLabel', strrep(samples,'Sample ',''), 'YTick', [], ...
    'TickLength', [0 0], 'FontWeight','bold','FontSize',7);
axH.XAxis.Color = 'none';  axH.XAxis.TickLabelColor = C.ink;  axH.YAxis.Color = 'none';
axH.XLabel.String = 'Sample (replicates)';  axH.XLabel.Color = C.ink;
axH.XLabel.FontWeight = 'normal';
ylabel(axH, sprintf('Top %d differentially abundant proteins', numel(top)), ...
    'Color', C.ink, 'FontSize', 7, 'FontWeight','normal');
axH.YLabel.Visible = 'on';
cb = colorbar(axH, 'Position', [0.91 0.30 0.012 0.20]);
cb.Label.String = 'z-score';  cb.FontSize = 6;  cb.Box = 'off';  cb.Color = C.ink2;
panelLetter(fig, axS, 'e', 0.05);

% ---- (f) Pathway dot plot ---------------------------------------------------
minQ = min(qEnr, [], 2);  [~, o] = sort(minQ);  nTop = min(8, sum(minQ < FDR_THRESH));
if nTop == 0, [~, o] = sort(min(pEnr,[],2)); nTop = min(8, nPW); end
sel = o(1:nTop);
ax = axes(fig, 'Position', [0.12 0.07 0.30 0.21]);  hold(ax,'on');
score = -log10(max(qEnr(sel,:), 1e-12));  vmax = max([score(:); 2]);
for r = 1:nTop
    yy = nTop + 1 - r;
    for s = 1:nS-1
        sz = (ovl(sel(r),s)*3 + 4) * 1.6;
        if qEnr(sel(r),s) < FDR_THRESH
            scatter(ax, s, yy, sz, score(r,s), 'filled', 'MarkerEdgeColor','w','LineWidth',0.5);
        else
            scatter(ax, s, yy, sz, 'MarkerEdgeColor', C.ns, 'LineWidth', 0.7);
        end
    end
end
colormap(ax, C.seq);  caxis(ax, [0 vmax]);
set(ax, 'XTick', 1:nS-1, 'XTickLabel', samples(2:end), 'YTick', 1:nTop, ...
    'YTickLabel', flipud(pwNames(sel)));
xlim(ax, [0.4 nS-0.4]);  ylim(ax, [0.3 nTop+0.7]);  styleAxes(ax, C);  grid(ax,'on');
title(ax, 'Pathway over-representation', 'FontWeight','normal','FontSize',7.5,'Color',C.ink);
cb2 = colorbar(ax, 'Position', [0.52 0.09 0.012 0.17]);
cb2.Label.String = '-log_{10}(FDR)';  cb2.FontSize = 6;  cb2.Box = 'off';  cb2.Color = C.ink2;
axL = axes(fig, 'Position', [0.44 0.09 0.06 0.17]);  hold(axL,'on');  axis(axL,'off');
lv = [10 30 60];
for i = 1:3
    scatter(axL, 0.3, 4-i, (lv(i)*3+4)*1.6, C.ink2, 'filled', 'MarkerEdgeColor','w');
    text(axL, 0.75, 4-i, num2str(lv(i)), 'FontSize', 6.5, 'Color', C.ink2);
end
text(axL, 0.3, 3.8, 'Proteins', 'FontSize', 6.5, 'Color', C.ink, 'VerticalAlignment','bottom');
xlim(axL, [0 1.4]);  ylim(axL, [0.4 4.2]);
panelLetter(fig, ax, 'f');

%% ---------------- EXPORT --------------------------------------------------
exportgraphics(fig, fullfile(OUT_DIR, 'graphical_abstract_proteomics_matlab.png'), 'Resolution', 600);
exportgraphics(fig, fullfile(OUT_DIR, 'graphical_abstract_proteomics_matlab.pdf'), 'ContentType', 'vector');
fprintf('Saved figure and tables to %s\n', OUT_DIR);

%% =================== LOCAL FUNCTIONS ======================================
function [X, proteins, colGroup, samples, pw] = simulateDataset(nP, nS, nR)
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    samples = arrayfun(@(i) sprintf('Sample %s', letters(i)), 1:nS, 'UniformOutput', false);
    proteins = arrayfun(@(i) sprintf('Protein_%04d', i), (1:nP)', 'UniformOutput', false);
    base = 22 + 2*randn(nP,1);
    nPW = 10;  pwIdx = cell(nPW,1);
    for k = 1:nPW, pwIdx{k} = randperm(nP, randi([60 159])); end
    eff = zeros(nP, nS);
    pool = unique([pwIdx{1} pwIdx{2}]);  core = pool(randperm(numel(pool), 110));
    coreFC = (2*(rand(110,1) < 0.55) - 1) .* (1.9 + 0.45*randn(110,1));
    for s = 2:nS, eff(core,s) = coreFC .* (1 + 0.15*randn(110,1)); end
    for s = 2:nS
        own = pwIdx{s+1};
        spec = unique([own(randperm(numel(own), min(70, numel(own)))) randperm(nP, 90)]);
        eff(spec,s) = eff(spec,s) + (2*(rand(numel(spec),1) < 0.5) - 1) .* (1.7 + 0.5*randn(numel(spec),1));
    end
    pair = pwIdx{6}(randperm(numel(pwIdx{6}), 45));
    for s = 2:min(3,nS), eff(pair,s) = eff(pair,s) - 1.8 + 0.4*randn(45,1); end
    X = zeros(nP, nS*nR);  colGroup = zeros(1, nS*nR);  c = 0;
    for s = 1:nS
        for r = 1:nR
            c = c + 1;
            X(:,c) = base + eff(:,s) + 0.35*randn(nP,1) + 0.05*randn;
            colGroup(c) = s;
        end
    end
    low = X < quantileSimple(X(:), 0.05);
    X(low & rand(size(X)) < 0.6) = NaN;           % MNAR
    X(rand(size(X)) < 0.01) = NaN;                % MCAR
    names = {};  prots = {};
    for k = 1:nPW
        names = [names; repmat({sprintf('Pathway %d', k)}, numel(pwIdx{k}), 1)]; %#ok<AGROW>
        prots = [prots; proteins(pwIdx{k})];                                  %#ok<AGROW>
    end
    pw = table(names, prots, 'VariableNames', {'Pathway','Protein'});
end

function [X, proteins, colGroup, samples, pw] = loadDataset(f, pf, refGroup, keepNames)
    T = readtable(f, 'VariableNamingRule', 'preserve');
    rawIDs = string(T{:,1});
    names = T.Properties.VariableNames(2:end);
    X = T{:, 2:end};
    grp = regexprep(names, '_R?\d+$', '');
    order = unique(grp, 'stable');
    if ~isempty(refGroup)
        if ~any(strcmp(order, refGroup)), error('REFERENCE_GROUP not found among column groups.'); end
        order = [{refGroup}, order(~strcmp(order, refGroup))];
    end
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    samples = arrayfun(@(i) sprintf('Sample %s', letters(i)), 1:numel(order), 'UniformOutput', false);
    colGroup = cellfun(@(g) find(strcmp(order, g)), grp);
    [colGroup, o] = sort(colGroup);  X = X(:, o);
    fprintf('Column groups relabeled Sample A..%s (reference first); original names are not used further.\n', ...
        letters(numel(order)));
    proteins = arrayfun(@(i) sprintf('Protein_%04d', i), (1:numel(rawIDs))', 'UniformOutput', false);
    if isempty(pf), error('PATHWAY_FILE is required when INPUT_FILE is set (panel f).'); end
    P = readtable(pf, 'VariableNamingRule', 'preserve');
    pwRaw = string(P{:,1});  prRaw = string(P{:,2});
    [ok, loc] = ismember(prRaw, rawIDs);
    pwRaw = pwRaw(ok);  prot = proteins(loc(ok));
    if ~keepNames
        u = unique(pwRaw, 'stable');
        [~, id] = ismember(pwRaw, u);
        pwRaw = arrayfun(@(i) sprintf('Pathway %d', i), id, 'UniformOutput', false);
    else
        pwRaw = cellstr(pwRaw);
    end
    pw = table(pwRaw(:), prot(:), 'VariableNames', {'Pathway','Protein'});
end

function [p, d] = welchT(a, b)
    n1 = size(a,2);  n2 = size(b,2);
    m1 = mean(a,2);  m2 = mean(b,2);  v1 = var(a,0,2);  v2 = var(b,0,2);
    se2 = v1/n1 + v2/n2;  t = (m1 - m2) ./ sqrt(se2);
    df = se2.^2 ./ ((v1/n1).^2/(n1-1) + (v2/n2).^2/(n2-1));
    p = betainc(df ./ (df + t.^2), df/2, 0.5);    % two-sided p-value
    d = m1 - m2;
end

function q = bhFDR(p)
    p = p(:);  n = numel(p);  [ps, o] = sort(p);
    qs = ps .* n ./ (1:n)';
    qs = flipud(cummin(flipud(qs)));
    q = zeros(n,1);  q(o) = min(qs, 1);
end

function p = hygeUpper(k, N, K, n)
    lnC = @(a,b) gammaln(a+1) - gammaln(b+1) - gammaln(a-b+1);
    i = k:min(K, n);
    p = min(1, sum(exp(lnC(K,i) + lnC(N-K, n-i) - lnC(N,n))));
end

function ord = clusterRows(Z)
    if exist('linkage', 'file') == 2 && exist('optimalleaforder', 'file') == 2
        D = pdist(Z, 'correlation');  L = linkage(D, 'average');
        ord = optimalleaforder(L, D);
    else                                           % toolbox-free fallback
        Zc = Z - mean(Z,1);  [U,S,~] = svd(Zc, 'econ');  sc = U(:,1:2)*S(1:2,1:2);
        [~, ord] = sort(atan2(sc(:,2), sc(:,1)));
    end
end

function v = quantileSimple(x, q)
    x = sort(x(~isnan(x)));  v = x(max(1, round(q*numel(x))));
end

function styleAxes(ax, C)
    ax.XColor = C.muted;  ax.YColor = C.muted;  ax.GridColor = C.grid;  ax.GridAlpha = 1;
    ax.XAxis.TickLabelColor = C.ink2;  ax.YAxis.TickLabelColor = C.ink2;  ax.Layer = 'bottom';
end

function padLimits(ax, f)
    xl = xlim(ax);  yl = ylim(ax);
    xlim(ax, xl + [-1 1]*f*diff(xl));  ylim(ax, yl + [-1 1]*f*diff(yl));
end

function panelLetter(fig, ax, letter, dy)
    if nargin < 4, dy = 0.035; end
    p = ax.Position;
    annotation(fig, 'textbox', [p(1)-0.06, p(2)+p(4)+dy-0.015, 0.03, 0.03], 'String', letter, ...
        'EdgeColor','none','FontWeight','bold','FontSize',9,'FontName','Arial', ...
        'VerticalAlignment','bottom');
end

function rgb = hex2rgb(h)
    h = char(h);  rgb = [hex2dec(h(2:3)) hex2dec(h(4:5)) hex2dec(h(6:7))] / 255;
end

function cmap = makeCmap(hexes)
    stops = cell2mat(cellfun(@hex2rgb, hexes, 'UniformOutput', false)');
    x = linspace(0, 1, size(stops,1));
    cmap = interp1(x, stops, linspace(0, 1, 256));
end
