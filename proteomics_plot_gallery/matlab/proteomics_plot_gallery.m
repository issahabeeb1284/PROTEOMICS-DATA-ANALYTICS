%% ANONYMIZED PROTEOMICS PLOT GALLERY: 28 publication-grade figures (MATLAB)
% =========================================================================
% Everything is anonymous: genotypes are "Sample A" (reference), "Sample B",
% "Sample C", "Sample D"; proteins are coded Protein_0001 ...; pathways are
% "Pathway 1" ... No real gene, protein or sample name appears anywhere.
%
% Needs only base MATLAB (R2020a or newer). No toolboxes: clustering,
% k-means, kernel density, box plots and violins are implemented below.
%
% QC & data structure                 Differential abundance
%  01 identifications per replicate     13 volcano plots
%  02 missing-value map                 14 MA plots
%  03 intensity boxplots raw vs norm.   15 p-value histograms
%  04 intensity density curves          16 log2FC distributions
%  05 imputation QC                     17 up/down counts
%  06 dynamic range                     18 UpSet overlap
%  07 CV distributions (violins)        19 log2FC concordance between samples
%  08 replicate correlation heatmap     20 clustered heatmap of top proteins
%  09 replicate dendrogram              21 k-means abundance-profile clusters
%  10 PCA scores                        22 top-protein box/strip plots
%  11 PCA scree                         23 ranked log2FC waterfall
%  12 PCA loadings
%                                      Pathway level
%                                       24 over-representation dot plot
%                                       25 enrichment bar chart
%                                       26 GSEA running enrichment score
%                                       27 pathway activity heatmap
%                                       28 pathway activity radar
%
% Output: one PNG (300 dpi) per plot + ALL_PLOTS_matlab.pdf
% To use real data, replace simulateDataset() with a loader returning a
% protein x replicate log2 matrix, the replicate-to-sample index and the
% pathway membership, after recoding every name to the anonymous scheme.
% =========================================================================
clear; close all; clc;

OUT = fullfile(fileparts(mfilename('fullpath')), 'output_plots');
if ~exist(OUT, 'dir'), mkdir(OUT); end
PDF = fullfile(OUT, 'ALL_PLOTS_matlab.pdf');
if exist(PDF, 'file'), delete(PDF); end

SEED = 42;  FDR_T = 0.05;  FC_T = 1.0;  MIN_VALID = 0.70;
rng(SEED);

%% Theme (validated palette; reference sample is neutral gray) ------------
C.ink = hex2rgb('#0b0b0b');  C.ink2 = hex2rgb('#52514e');  C.muted = hex2rgb('#898781');
C.grid = hex2rgb('#e1e0d9'); C.ns = hex2rgb('#c3c2b7');    C.up = hex2rgb('#e34948');
C.down = hex2rgb('#2a78d6'); C.light = hex2rgb('#f0efec');
C.div = makeCmap({'#184f95','#2a78d6','#f0efec','#e34948','#8a2323'});
C.seq = makeCmap({'#b5d3f7','#6da7ec','#2a78d6','#184f95'});
COL = [hex2rgb('#52514e'); hex2rgb('#2a78d6'); hex2rgb('#1baf7a'); hex2rgb('#4a3aa7'); ...
       hex2rgb('#eda100'); hex2rgb('#e87ba4')];

%% Data: simulated anonymous LFQ experiment --------------------------------
[raw, proteins, grp, samples, pwIdx, pwNames] = simulateDataset(3000, 4, 4);
nS = numel(samples);  nO = nS - 1;  others = samples(2:end);
nRep = sum(grp == 1);
short = arrayfun(@(j) sprintf('%s-R%d', samples{grp(j)}(end), sum(grp(1:j) == grp(j))), ...
                 1:numel(grp), 'UniformOutput', false);
letters = cellfun(@(s) s(end), samples, 'UniformOutput', false);

% Preprocess: valid-value filter -> median normalization -> imputation
vf = zeros(size(raw,1), nS);
for s = 1:nS, vf(:,s) = mean(~isnan(raw(:, grp == s)), 2); end
keep = any(vf >= MIN_VALID, 2);
rawK = raw(keep,:);  proteins = proteins(keep);  nP = sum(keep);
med = median(rawK, 1, 'omitnan');
NORM = rawK - med + mean(med);
IMP = isnan(NORM);
X = NORM;
for j = 1:size(X,2)
    mu = mean(NORM(:,j), 'omitnan');  sd = std(NORM(:,j), 'omitnan');
    X(IMP(:,j), j) = (mu - 1.8*sd) + 0.3*sd*randn(sum(IMP(:,j)), 1);
end
memb = false(nP, numel(pwNames));
keptIdx = find(keep);
for k = 1:numel(pwNames), memb(:,k) = ismember(keptIdx, pwIdx{k}); end
fprintf('%d proteins simulated, %d retained\n', size(raw,1), nP);

% Differential abundance: Welch t-test vs Sample A + BH FDR
L2FC = zeros(nP, nO);  P = L2FC;  Q = L2FC;  T = L2FC;  AM = L2FC;  CALL = zeros(nP, nO);
ref = X(:, grp == 1);
for s = 1:nO
    tst = X(:, grp == s + 1);
    [P(:,s), L2FC(:,s), T(:,s)] = welchT(tst, ref);
    AM(:,s) = (mean(tst,2) + mean(ref,2)) / 2;
    Q(:,s) = bhFDR(P(:,s));
    CALL(Q(:,s) < FDR_T & L2FC(:,s) >  FC_T, s) =  1;
    CALL(Q(:,s) < FDR_T & L2FC(:,s) < -FC_T, s) = -1;
end
NLQ = -log10(max(Q, 1e-300));

% Pathway over-representation (hypergeometric) + BH
nPW = numel(pwNames);  ovl = zeros(nPW, nO);  pE = ones(nPW, nO);  qE = pE;
for s = 1:nO
    hit = CALL(:,s) ~= 0;  n = sum(hit);
    for k = 1:nPW
        K = sum(memb(:,k));  kk = sum(hit & memb(:,k));  ovl(k,s) = kk;
        if kk > 0, pE(k,s) = hygeUpper(kk, nP, K, n); end
    end
    qE(:,s) = bhFDR(pE(:,s));
end

%% 01 Protein identifications per replicate -------------------------------
fig = newFig(89, 62);  ax = axes(fig);  hold(ax, 'on');
ids = sum(~isnan(raw), 1);
b = bar(ax, ids, 0.72, 'FaceColor', 'flat', 'EdgeColor', 'none');
b.CData = COL(grp, :);
ylim(ax, [min(ids)*0.97, max(ids)*1.01]);
set(ax, 'XTick', 1:numel(ids), 'XTickLabel', short, 'XTickLabelRotation', 90);
ylabel(ax, 'Quantified proteins');  title(ax, 'Protein identifications per replicate');
sampleLegend(ax, samples, COL, 'southoutside', 'horizontal');
finishAxes(ax, C);  saveFig(fig, OUT, PDF, '01_identifications_per_replicate');

%% 02 Missing-value map ----------------------------------------------------
fig = newFig(89, 75);  ax = axes(fig);
mrows = find(any(isnan(raw), 2));
[~, o] = sort(mean(raw(mrows,:), 2, 'omitnan'));
imagesc(ax, double(isnan(raw(mrows(o), :))));
colormap(ax, [C.light; C.ink2]);
set(ax, 'XTick', 1:numel(short), 'XTickLabel', short, 'XTickLabelRotation', 90, 'YTick', []);
ylabel(ax, {sprintf('%d proteins with \\geq1 missing value', numel(mrows)), ...
            '(sorted by mean abundance, low \rightarrow high)'});
title(ax, sprintf('Missing-value map (%.1f%% missing overall)', 100*mean(isnan(raw(:)))));
hold(ax, 'on');
h1 = patch(ax, NaN, NaN, C.ink2, 'EdgeColor', 'none');
h2 = patch(ax, NaN, NaN, C.light, 'EdgeColor', C.muted);
legend(ax, [h1 h2], {'Missing', 'Observed'}, 'Location', 'southoutside', ...
       'Orientation', 'horizontal', 'Box', 'off');
finishAxes(ax, C);  saveFig(fig, OUT, PDF, '02_missing_value_map');

%% 03 Intensity boxplots: raw vs normalized --------------------------------
fig = newFig(130, 62);  tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
data = {rawK, NORM};  ttl = {'Before normalization', 'After median normalization'};
for t = 1:2
    ax = nexttile(tl);  hold(ax, 'on');
    for j = 1:size(rawK,2)
        drawBox(ax, j, data{t}(:,j), COL(grp(j),:), 0.6, 1, C);
    end
    set(ax, 'XTick', 1:numel(short), 'XTickLabel', short, 'XTickLabelRotation', 90);
    xlim(ax, [0.3 numel(short)+0.7]);  title(ax, ttl{t});
    if t == 1, ylabel(ax, 'log_2 intensity'); else, ax.YTickLabel = []; end
    finishAxes(ax, C);
end
linkaxes(tl.Children, 'y');
saveFig(fig, OUT, PDF, '03_intensity_boxplots_raw_vs_normalized');

%% 04 Intensity density curves ---------------------------------------------
fig = newFig(89, 62);  ax = axes(fig);  hold(ax, 'on');
g = linspace(min(NORM(:)), max(NORM(:)), 300);
for j = 1:size(NORM,2)
    plot(ax, g, kde(NORM(~isnan(NORM(:,j)), j), g), 'Color', [COL(grp(j),:) 0.85], 'LineWidth', 1);
end
xlabel(ax, 'log_2 intensity (normalized)');  ylabel(ax, 'Density');
title(ax, 'Intensity distributions per replicate');
sampleLegend(ax, samples, COL, 'northeast', 'vertical');
finishAxes(ax, C);  saveFig(fig, OUT, PDF, '04_intensity_density');

%% 05 Imputation QC ------------------------------------------------------------
fig = newFig(89, 62);  ax = axes(fig);  hold(ax, 'on');
edges = linspace(min(X(:)), max(X(:)), 60);
histogram(ax, X(~IMP), edges, 'FaceColor', C.ns, 'EdgeColor', 'none', 'FaceAlpha', 1);
histogram(ax, X(IMP), edges, 'FaceColor', C.down, 'EdgeColor', 'none', 'FaceAlpha', 1);
legend(ax, {sprintf('Observed (n = %d)', sum(~IMP(:))), sprintf('Imputed (n = %d)', sum(IMP(:)))}, ...
       'Box', 'off', 'Location', 'northeast');
xlabel(ax, 'log_2 intensity');  ylabel(ax, 'Values');
title(ax, 'Left-shifted imputation of missing values');
finishAxes(ax, C);  saveFig(fig, OUT, PDF, '05_imputation_qc');

%% 06 Dynamic range --------------------------------------------------------------
fig = newFig(89, 62);  ax = axes(fig);  hold(ax, 'on');
for s = 1:nS
    m = sort(mean(X(:, grp == s), 2), 'descend');
    plot(ax, 1:nP, m, 'Color', COL(s,:), 'LineWidth', 1.4);
end
rm = mean(X(:, grp == 1), 2);
span = qtile(rm, 0.99) - qtile(rm, 0.01);
text(ax, 0.97, 0.92, sprintf('\\approx%.1f orders of magnitude\n(1st-99th percentile)', span/log2(10)), ...
     'Units', 'normalized', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'Color', C.ink2);
xlabel(ax, 'Abundance rank');  ylabel(ax, 'Mean log_2 intensity');
title(ax, 'Dynamic range of the quantified proteome');
legend(ax, samples, 'Box', 'off', 'Location', 'southwest');
finishAxes(ax, C);  saveFig(fig, OUT, PDF, '06_dynamic_range');

%% 07 CV distributions (violins) -------------------------------------------------
fig = newFig(89, 66);  ax = axes(fig);  hold(ax, 'on');
for s = 1:nS
    lin = 2.^NORM(:, grp == s);  lin = lin(all(~isnan(lin), 2), :);
    cv = min(std(lin, 0, 2) ./ mean(lin, 2) * 100, 100);
    drawViolin(ax, s, cv, COL(s,:), 0.4);
    md = median(cv);
    plot(ax, s + [-0.18 0.18], [md md], 'Color', 'w', 'LineWidth', 1.6);
    text(ax, s, 104, sprintf('median\n%.1f%%', md), 'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'bottom', 'FontSize', 6, 'Color', C.ink2);
end
yline(ax, 20, '--', 'Color', C.muted, 'LineWidth', 0.6);
set(ax, 'XTick', 1:nS, 'XTickLabel', samples, 'YTick', 0:20:100);
xlim(ax, [0.4 nS+0.6]);  ylim(ax, [0 120]);
ylabel(ax, 'Coefficient of variation (%)');
title(ax, 'Replicate precision (dashed line = 20% CV)');
finishAxes(ax, C);  saveFig(fig, OUT, PDF, '07_cv_distribution');

%% 08 Replicate correlation heatmap -----------------------------------------------
fig = newFig(89, 82);  ax = axes(fig);
R = corrcoef(X);
imagesc(ax, R, [floor(min(R(:))*100)/100 1]);  colormap(ax, C.seq);  axis(ax, 'square');
lbl = colorLabels(short, grp, COL);
set(ax, 'XTick', 1:numel(short), 'XTickLabel', lbl, 'YTick', 1:numel(short), 'YTickLabel', lbl, ...
    'XTickLabelRotation', 90, 'TickLabelInterpreter', 'tex', 'TickLength', [0 0]);
cb = colorbar(ax);  cb.Label.String = 'Pearson r';  cb.Box = 'off';
title(ax, 'Replicate-to-replicate correlation');
finishAxes(ax, C);  hideAxisLines(ax, C, true, true);
saveFig(fig, OUT, PDF, '08_replicate_correlation_heatmap');

%% 09 Replicate dendrogram ------------------------------------------------------------
fig = newFig(89, 62);  ax = axes(fig);  hold(ax, 'on');
Zl = upgma(1 - corrcoef(X));
ord = drawDendrogram(ax, Zl, size(X,2), C.ink2);
set(ax, 'XTick', 1:numel(ord), 'XTickLabel', colorLabels(short(ord), grp(ord), COL), ...
    'XTickLabelRotation', 90, 'TickLabelInterpreter', 'tex', 'TickLength', [0 0]);
xlim(ax, [0.3 numel(ord)+0.7]);
ylabel(ax, '1 - Pearson r');  title(ax, 'Unsupervised clustering of replicates');
finishAxes(ax, C);  hideAxisLines(ax, C, true, false);
saveFig(fig, OUT, PDF, '09_replicate_dendrogram');

%% 10-12 PCA ---------------------------------------------------------------------------
Xc = X' - mean(X', 1);
[U, S, V] = svd(Xc, 'econ');  PCS = U*S;  ev = diag(S).^2;  VAR = 100*ev/sum(ev);

fig = newFig(89, 72);  ax = axes(fig);  hold(ax, 'on');  grid(ax, 'on');
hs = gobjects(nS,1);
for s = 1:nS
    m = grp == s;
    hs(s) = scatter(ax, PCS(m,1), PCS(m,2), 40, COL(s,:), 'filled', 'MarkerEdgeColor', 'w');
end
yr = range2(PCS(:,2));
for s = 1:nS
    m = grp == s;
    text(ax, mean(PCS(m,1)), max(PCS(m,2)) + 0.07*yr, letters{s}, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'Color', C.ink);
end
padLimits(ax, 0.15);
xlabel(ax, sprintf('PC1 (%.1f%%)', VAR(1)));  ylabel(ax, sprintf('PC2 (%.1f%%)', VAR(2)));
title(ax, 'Principal component analysis');
legend(ax, hs, samples, 'Box', 'off', 'Location', 'eastoutside');
finishAxes(ax, C);  saveFig(fig, OUT, PDF, '10_pca_scores');

fig = newFig(89, 58);  ax = axes(fig);  hold(ax, 'on');
k = min(10, numel(VAR));
bar(ax, 1:k, VAR(1:k), 0.65, 'FaceColor', C.ink2, 'EdgeColor', 'none');
plot(ax, 1:k, cumsum(VAR(1:k)), '-o', 'Color', C.down, 'MarkerFaceColor', C.down, ...
     'MarkerSize', 4, 'LineWidth', 1.4);
set(ax, 'XTick', 1:k, 'XTickLabel', arrayfun(@(i) sprintf('PC%d', i), 1:k, 'UniformOutput', false));
ylim(ax, [0 105]);  ylabel(ax, 'Variance explained (%)');  title(ax, 'PCA scree plot');
legend(ax, {'Per component', 'Cumulative'}, 'Box', 'off', 'Location', 'east');
finishAxes(ax, C);  saveFig(fig, OUT, PDF, '11_pca_scree');

fig = newFig(89, 72);  ax = axes(fig);  hold(ax, 'on');
ld = V(:, 1:2);  mag = hypot(ld(:,1), ld(:,2));  top = mag >= qtile(mag, 0.98);
scatter(ax, ld(~top,1), ld(~top,2), 4, C.ns, 'filled');
scatter(ax, ld(top,1), ld(top,2), 9, C.down, 'filled');
xline(ax, 0, 'Color', C.muted, 'LineWidth', 0.5);  yline(ax, 0, 'Color', C.muted, 'LineWidth', 0.5);
xlabel(ax, 'PC1 loading');  ylabel(ax, 'PC2 loading');
title(ax, 'PCA loadings: proteins driving separation');
legend(ax, {'All proteins', 'Top 2% contributors'}, 'Box', 'off', 'Location', 'northwest');
finishAxes(ax, C);  saveFig(fig, OUT, PDF, '12_pca_loadings');

%% 13 Volcano plots -----------------------------------------------------------------------
xl = ceil(qtile(abs(L2FC(:)), 0.999) + 0.5);  yl = qtile(NLQ(:), 0.999) * 1.12;
fig = newFig(183, 62);  tl = tiledlayout(fig, 1, nO, 'TileSpacing', 'compact', 'Padding', 'compact');
for s = 1:nO
    ax = nexttile(tl);  hold(ax, 'on');
    scatterCalls(ax, L2FC(:,s), NLQ(:,s), CALL(:,s), C);
    xline(ax, [-FC_T FC_T], '--', 'Color', C.muted, 'LineWidth', 0.5);
    yline(ax, -log10(FDR_T), '--', 'Color', C.muted, 'LineWidth', 0.5);
    xlim(ax, [-xl xl]);  ylim(ax, [0 yl]);
    text(ax, 0.04, 0.97, sprintf('\\nabla %d', sum(CALL(:,s) == -1)), 'Units', 'normalized', ...
         'VerticalAlignment', 'top', 'Color', C.ink2);
    text(ax, 0.96, 0.97, sprintf('%d \\Delta', sum(CALL(:,s) == 1)), 'Units', 'normalized', ...
         'VerticalAlignment', 'top', 'HorizontalAlignment', 'right', 'Color', C.ink2);
    title(ax, sprintf('%s vs Sample A', others{s}));  xlabel(ax, 'log_2 fold change');
    if s == 1, ylabel(ax, '-log_{10}(FDR)'); else, ax.YTickLabel = []; end
    finishAxes(ax, C);
end
saveFig(fig, OUT, PDF, '13_volcano_plots');

%% 14 MA plots --------------------------------------------------------------------------
fig = newFig(183, 60);  tl = tiledlayout(fig, 1, nO, 'TileSpacing', 'compact', 'Padding', 'compact');
for s = 1:nO
    ax = nexttile(tl);  hold(ax, 'on');
    scatterCalls(ax, AM(:,s), L2FC(:,s), CALL(:,s), C);
    yline(ax, 0, 'Color', C.ink2, 'LineWidth', 0.6);
    yline(ax, [-FC_T FC_T], '--', 'Color', C.muted, 'LineWidth', 0.5);
    title(ax, sprintf('%s vs Sample A', others{s}));  xlabel(ax, 'Mean log_2 intensity (A)');
    if s == 1, ylabel(ax, 'log_2 fold change (M)'); else, ax.YTickLabel = []; end
    finishAxes(ax, C);
end
linkaxes(tl.Children, 'y');
saveFig(fig, OUT, PDF, '14_ma_plots');

%% 15 p-value histograms ----------------------------------------------------------------
fig = newFig(183, 52);  tl = tiledlayout(fig, 1, nO, 'TileSpacing', 'compact', 'Padding', 'compact');
for s = 1:nO
    ax = nexttile(tl);  hold(ax, 'on');
    histogram(ax, P(:,s), linspace(0, 1, 41), 'FaceColor', COL(s+1,:), 'EdgeColor', 'none', 'FaceAlpha', 1);
    yline(ax, nP/40, '--', 'Color', C.ink2, 'LineWidth', 0.6);
    title(ax, sprintf('%s vs Sample A', others{s}));  xlabel(ax, 'Raw p-value');
    if s == 1, ylabel(ax, 'Proteins'); end
    finishAxes(ax, C);
end
linkaxes(tl.Children, 'y');
text(ax, 1, nP/40, 'uniform null', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', ...
     'FontSize', 6, 'Color', C.ink2);
saveFig(fig, OUT, PDF, '15_pvalue_histograms');

%% 16 log2FC distributions ----------------------------------------------------------------
fig = newFig(89, 62);  ax = axes(fig);  hold(ax, 'on');
for s = 1:nO
    histogram(ax, L2FC(:,s), linspace(-5, 5, 61), 'DisplayStyle', 'stairs', ...
              'EdgeColor', COL(s+1,:), 'LineWidth', 1.3);
end
set(ax, 'YScale', 'log');  ylim(ax, [0.8 Inf]);
xline(ax, 0, 'Color', C.ink2, 'LineWidth', 0.6);
xline(ax, [-FC_T FC_T], '--', 'Color', C.muted, 'LineWidth', 0.5);
xlabel(ax, 'log_2 fold change vs Sample A');  ylabel(ax, 'Proteins (log scale)');
title(ax, 'Fold-change distributions');
legend(ax, others, 'Box', 'off', 'Location', 'northeast');
finishAxes(ax, C);  saveFig(fig, OUT, PDF, '16_log2fc_distributions');

%% 17 Up/down counts -------------------------------------------------------------------------
fig = newFig(89, 58);  ax = axes(fig);  hold(ax, 'on');
nUp = sum(CALL == 1, 1);  nDn = sum(CALL == -1, 1);  y = nO:-1:1;  mx = max([nUp nDn]);
hU = barh(ax, y, nUp, 0.6, 'FaceColor', C.up, 'EdgeColor', 'none');
hD = barh(ax, y, -nDn, 0.6, 'FaceColor', C.down, 'EdgeColor', 'none');
for i = 1:nO
    text(ax, nUp(i) + 0.03*mx, y(i), num2str(nUp(i)), 'Color', C.ink2);
    text(ax, -nDn(i) - 0.03*mx, y(i), num2str(nDn(i)), 'Color', C.ink2, 'HorizontalAlignment', 'right');
end
xline(ax, 0, 'Color', C.ink2, 'LineWidth', 0.6);
xlim(ax, [-1.3 1.3]*mx);  ylim(ax, [0.4 nO+0.6]);
set(ax, 'YTick', fliplr(y), 'YTickLabel', fliplr(others), 'TickLength', [0 0]);
ax.XTickLabel = arrayfun(@(v) num2str(abs(v)), ax.XTick, 'UniformOutput', false);
xlabel(ax, 'Differentially abundant proteins');  title(ax, 'Direction of change');
legend(ax, [hU hD], {'Higher than Sample A', 'Lower than Sample A'}, 'Box', 'off', ...
       'Location', 'southoutside', 'Orientation', 'horizontal');
finishAxes(ax, C);  ax.YAxis.Color = 'none';  ax.YAxis.TickLabelColor = C.ink2;
saveFig(fig, OUT, PDF, '17_up_down_counts');

%% 18 UpSet overlap ----------------------------------------------------------------------------
sigM = CALL ~= 0;  combos = [];  cnt = [];
for mask = 1:(2^nO - 1)
    inS = bitget(mask, 1:nO) == 1;
    ex = all(sigM(:, inS), 2) & ~any(sigM(:, ~inS), 2);
    if any(ex), combos = [combos; inS]; cnt = [cnt; sum(ex)]; end %#ok<AGROW>
end
[cnt, o] = sort(cnt, 'descend');  combos = combos(o,:);  nc = numel(cnt);
fig = newFig(89, 72);  tl = tiledlayout(fig, 3, 1, 'TileSpacing', 'none', 'Padding', 'compact');
ax = nexttile(tl, [2 1]);  hold(ax, 'on');
bar(ax, 1:nc, cnt, 0.6, 'FaceColor', C.ink2, 'EdgeColor', 'none');
for i = 1:nc
    text(ax, i, cnt(i), num2str(cnt(i)), 'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'bottom', 'FontSize', 6, 'Color', C.ink2);
end
xlim(ax, [0.4 nc+0.6]);  ylim(ax, [0 max(cnt)*1.15]);
ylabel(ax, 'Proteins');  title(ax, 'Shared and unique changes (UpSet)');
finishAxes(ax, C);  ax.XAxis.Visible = 'off';
ax = nexttile(tl);  hold(ax, 'on');
for i = 1:nc
    yy = nO + 1 - find(combos(i,:));
    if numel(yy) > 1, plot(ax, [i i], [min(yy) max(yy)], '-', 'Color', C.ink2, 'LineWidth', 1); end
    for s = 1:nO
        if combos(i,s), c = COL(s+1,:); sz = 26; else, c = C.grid; sz = 16; end
        scatter(ax, i, nO + 1 - s, sz, c, 'filled');
    end
end
xlim(ax, [0.4 nc+0.6]);  ylim(ax, [0.4 nO+0.6]);
set(ax, 'YTick', 1:nO, 'YTickLabel', fliplr(others), 'TickLength', [0 0]);
finishAxes(ax, C);  ax.XAxis.Visible = 'off';  ax.YAxis.Color = 'none';  ax.YAxis.TickLabelColor = C.ink2;
saveFig(fig, OUT, PDF, '18_upset_overlap');

%% 19 log2FC concordance ------------------------------------------------------------------------
pairs = nchoosek(1:nO, 2);  sigAny = any(CALL ~= 0, 2);
lim = ceil(max(arrayfun(@(s) qtile(abs(L2FC(:,s)), 0.999), 1:nO)));
fig = newFig(183, 64);  tl = tiledlayout(fig, 1, size(pairs,1), 'TileSpacing', 'compact', 'Padding', 'compact');
for p = 1:size(pairs,1)
    a = pairs(p,1);  bb = pairs(p,2);
    ax = nexttile(tl);  hold(ax, 'on');
    scatter(ax, L2FC(~sigAny,a), L2FC(~sigAny,bb), 3, C.ns, 'filled');
    scatter(ax, L2FC(sigAny,a), L2FC(sigAny,bb), 6, C.ink2, 'filled', 'MarkerFaceAlpha', 0.8);
    plot(ax, [-lim lim], [-lim lim], '--', 'Color', C.muted, 'LineWidth', 0.5);
    r = corrcoef(L2FC(:,a), L2FC(:,bb));
    text(ax, 0.04, 0.96, sprintf('r = %.2f', r(1,2)), 'Units', 'normalized', ...
         'VerticalAlignment', 'top', 'Color', C.ink2);
    xlim(ax, [-lim lim]);  ylim(ax, [-lim lim]);  axis(ax, 'square');
    xlabel(ax, ['log_2FC ' others{a}]);  ylabel(ax, ['log_2FC ' others{bb}]);
    finishAxes(ax, C);
end
title(tl, 'Concordance of fold changes between samples (dark = significant in any sample)', ...
      'FontWeight', 'bold', 'FontSize', 8);
saveFig(fig, OUT, PDF, '19_log2fc_concordance');

%% 20 Clustered heatmap of top proteins ------------------------------------------------------------
top = [];
for s = 1:nO
    idx = find(CALL(:,s) ~= 0);  [~, o] = sort(Q(idx,s));  idx = idx(o);
    top = [top; idx(1:min(20, numel(idx)))]; %#ok<AGROW>
end
top = unique(top, 'stable');
Zm = X(top,:);  Zm = (Zm - mean(Zm,2)) ./ std(Zm, 0, 2);
[~, ro] = upgma(1 - corrcoef(Zm'));  Zm = Zm(ro,:);
fig = newFig(89, 115);  ax = axes(fig);
imagesc(ax, Zm, [-2.5 2.5]);  colormap(ax, C.div);  hold(ax, 'on');
bounds = cumsum(arrayfun(@(s) sum(grp == s), 1:nS));
for bd = bounds(1:end-1), xline(ax, bd + 0.5, 'Color', 'w', 'LineWidth', 1.5); end
set(ax, 'XTick', 1:numel(short), 'XTickLabel', colorLabels(short, grp, COL), 'XTickLabelRotation', 90, ...
    'YTick', 1:numel(top), 'YTickLabel', proteins(top(ro)), 'TickLabelInterpreter', 'tex', ...
    'TickLength', [0 0]);
ax.YAxis.FontSize = 4.5;
cb = colorbar(ax);  cb.Label.String = 'Row z-score';  cb.Box = 'off';
title(ax, sprintf('Top %d differentially abundant proteins', numel(top)));
finishAxes(ax, C);  hideAxisLines(ax, C, true, true);
ax.YAxis.TickLabelInterpreter = 'none';
saveFig(fig, OUT, PDF, '20_top_protein_heatmap');

%% 21 k-means abundance-profile clusters --------------------------------------------------------------
sig = find(any(CALL ~= 0, 2));
mp = zeros(numel(sig), nS);
for s = 1:nS, mp(:,s) = mean(X(sig, grp == s), 2); end
zp = (mp - mean(mp,2)) ./ std(mp, 0, 2);
lab = kmeansPP(zp, 4, 200);
fig = newFig(183, 55);  tl = tiledlayout(fig, 1, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
for kk = 1:4
    ax = nexttile(tl);  hold(ax, 'on');  grid(ax, 'on');  ax.XGrid = 'off';
    mem = zp(lab == kk, :);
    plot(ax, 1:nS, mem', 'Color', [C.ns 0.6], 'LineWidth', 0.4);
    plot(ax, 1:nS, mean(mem, 1), '-o', 'Color', C.ink, 'MarkerFaceColor', C.ink, ...
         'MarkerSize', 4, 'LineWidth', 2);
    set(ax, 'XTick', 1:nS, 'XTickLabel', letters);  xlim(ax, [0.7 nS+0.3]);
    title(ax, sprintf('Cluster %d (n = %d)', kk, size(mem,1)));
    if kk == 1, ylabel(ax, 'Abundance z-score'); else, ax.YTickLabel = []; end
    finishAxes(ax, C);
end
linkaxes(tl.Children, 'y');
title(tl, 'Co-regulated protein clusters (k-means on differentially abundant proteins)', ...
      'FontWeight', 'bold', 'FontSize', 8);
saveFig(fig, OUT, PDF, '21_kmeans_profile_clusters');

%% 22 Top-protein box/strip plots ----------------------------------------------------------------------
[~, o] = sort(min(Q, [], 2));  best = o(1:6);
fig = newFig(130, 85);  tl = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:6
    ax = nexttile(tl);  hold(ax, 'on');
    for s = 1:nS
        v = X(best(i), grp == s)';
        drawBox(ax, s, v, COL(s,:), 0.55, 0.25, C);
        scatter(ax, s + (rand(size(v)) - 0.5)*0.24, v, 14, COL(s,:), 'filled', 'MarkerEdgeColor', 'w');
    end
    set(ax, 'XTick', 1:nS, 'XTickLabel', letters);  xlim(ax, [0.4 nS+0.6]);
    title(ax, proteins{best(i)}, 'Interpreter', 'none', 'FontSize', 7);
    if mod(i, 3) == 1, ylabel(ax, 'log_2 intensity'); end
    finishAxes(ax, C);
end
title(tl, 'Most significant proteins across samples', 'FontWeight', 'bold', 'FontSize', 8);
saveFig(fig, OUT, PDF, '22_top_protein_boxplots');

%% 23 Ranked log2FC waterfall --------------------------------------------------------------------------
fig = newFig(89, 88);  tl = tiledlayout(fig, nO, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
for s = 1:nO
    ax = nexttile(tl);  hold(ax, 'on');
    [v, o] = sort(L2FC(:,s), 'descend');  cl = CALL(o,s);
    cdat = repmat(C.ns, nP, 1);  cdat(cl == 1, :) = repmat(C.up, sum(cl == 1), 1);
    cdat(cl == -1, :) = repmat(C.down, sum(cl == -1), 1);
    b = bar(ax, 1:nP, v, 1, 'FaceColor', 'flat', 'EdgeColor', 'none');  b.CData = cdat;
    yline(ax, 0, 'Color', C.ink2, 'LineWidth', 0.5);
    text(ax, 0.99, 0.92, others{s}, 'Units', 'normalized', 'HorizontalAlignment', 'right', ...
         'VerticalAlignment', 'top', 'FontWeight', 'bold');
    ylabel(ax, 'log_2FC');  xlim(ax, [0 nP+1]);
    if s == 1, title(ax, 'Ranked fold-change waterfall'); end
    if s < nO, ax.XTickLabel = []; else, xlabel(ax, 'Proteins ranked by fold change'); end
    finishAxes(ax, C);
end
linkaxes(tl.Children, 'y');
saveFig(fig, OUT, PDF, '23_ranked_log2fc_waterfall');

%% 24 Pathway over-representation dot plot ---------------------------------------------------------------
[~, o] = sort(min(qE, [], 2));  topPW = o(1:min(10, nPW));  nT = numel(topPW);
score = -log10(max(qE(topPW,:), 1e-12));
fig = newFig(89, 82);  ax = axes(fig);  hold(ax, 'on');  grid(ax, 'on');
for r = 1:nT
    yy = nT + 1 - r;
    for s = 1:nO
        sz = (ovl(topPW(r),s)*3 + 5) * 1.5;
        if qE(topPW(r),s) < FDR_T
            scatter(ax, s, yy, sz, score(r,s), 'filled', 'MarkerEdgeColor', 'w');
        else
            scatter(ax, s, yy, sz, 'MarkerEdgeColor', C.ns, 'LineWidth', 0.7);
        end
    end
end
colormap(ax, C.seq);  caxis(ax, [0 max(score(:))]);
cb = colorbar(ax);  cb.Label.String = '-log_{10}(FDR)';  cb.Box = 'off';
hl = gobjects(3,1);  lv = [10 30 60];
for i = 1:3
    hl(i) = scatter(ax, NaN, NaN, (lv(i)*3 + 5)*1.5, C.ink2, 'filled', 'MarkerEdgeColor', 'w');
end
legend(ax, hl, {'10', '30', '60'}, 'Box', 'off', 'Location', 'southoutside', 'Orientation', 'horizontal');
set(ax, 'XTick', 1:nO, 'XTickLabel', others, 'YTick', 1:nT, 'YTickLabel', flipud(pwNames(topPW)));
xlim(ax, [0.4 nO+0.6]);  ylim(ax, [0.4 nT+0.6]);
title(ax, 'Pathway over-representation (dot size = proteins)');
finishAxes(ax, C);  saveFig(fig, OUT, PDF, '24_pathway_dotplot');

%% 25 Enrichment bar chart --------------------------------------------------------------------------------
fig = newFig(89, 78);  ax = axes(fig);  hold(ax, 'on');
M = -log10(max(qE(flipud(topPW), :), 1e-12));
b = barh(ax, 1:nT, M, 'grouped', 'EdgeColor', 'none');
for s = 1:nO, b(s).FaceColor = COL(s+1,:); end
xline(ax, -log10(FDR_T), '--', 'Color', C.ink2, 'LineWidth', 0.6);
text(ax, -log10(FDR_T), nT + 0.6, ' FDR 0.05', 'Color', C.ink2, 'FontSize', 6, ...
     'VerticalAlignment', 'bottom');
ylim(ax, [0.4 nT + 1.1]);
set(ax, 'YTick', 1:nT, 'YTickLabel', pwNames(flipud(topPW)), 'TickLength', [0 0]);
xlabel(ax, '-log_{10}(FDR)');  title(ax, 'Pathway enrichment by sample');
legend(ax, b, others, 'Box', 'off', 'Location', 'southeast');
finishAxes(ax, C);  saveFig(fig, OUT, PDF, '25_pathway_enrichment_bars');

%% 26 GSEA running enrichment score ------------------------------------------------------------------------
sG = 1;  [~, bp] = min(qE(:, sG));
[tr, o] = sort(T(:, sG), 'descend');  hits = memb(o, bp);
w = abs(tr) .* hits;  es = cumsum(w)/sum(w) - cumsum(~hits)/sum(~hits);
[~, pk] = max(abs(es));
fig = newFig(89, 80);  tl = tiledlayout(fig, 6, 1, 'TileSpacing', 'none', 'Padding', 'compact');
ax = nexttile(tl, [3 1]);  hold(ax, 'on');
plot(ax, 1:nP, es, 'Color', COL(sG+1,:), 'LineWidth', 1.4);
yline(ax, 0, 'Color', C.ink2, 'LineWidth', 0.5);  xline(ax, pk, '--', 'Color', C.muted, 'LineWidth', 0.5);
if pk > nP/2, ha = 'right'; else, ha = 'left'; end
text(ax, pk, es(pk), sprintf('  ES = %.2f  ', es(pk)), 'HorizontalAlignment', ha, 'Color', C.ink2);
ylabel(ax, 'Running ES');  xlim(ax, [1 nP]);
title(ax, sprintf('GSEA: %s, %s vs Sample A', pwNames{bp}, others{sG}));
finishAxes(ax, C);  ax.XAxis.Visible = 'off';
ax = nexttile(tl);  hold(ax, 'on');
hx = find(hits);  plot(ax, [hx hx]', repmat([0; 1], 1, numel(hx)), 'Color', C.ink, 'LineWidth', 0.4);
xlim(ax, [1 nP]);  axis(ax, 'off');
ax = nexttile(tl, [2 1]);  hold(ax, 'on');
area(ax, 1:nP, tr, 'FaceColor', C.ns, 'EdgeColor', 'none');
yline(ax, 0, 'Color', C.ink2, 'LineWidth', 0.5);
xlim(ax, [1 nP]);  ylabel(ax, 't statistic');  xlabel(ax, 'Protein rank');
finishAxes(ax, C);  saveFig(fig, OUT, PDF, '26_gsea_running_score');

%% 27 Pathway activity heatmap --------------------------------------------------------------------------------
zall = (X - mean(X,2)) ./ std(X, 0, 2);
act = zeros(nT, size(X,2));
for r = 1:nT, act(r,:) = mean(zall(memb(:, topPW(r)), :), 1); end
[~, ao] = upgma(squareDist(act));  act = act(ao,:);  actNames = pwNames(topPW(ao));
fig = newFig(89, 70);  ax = axes(fig);
lim = max(abs(act(:)));
imagesc(ax, act, [-lim lim]);  colormap(ax, C.div);  hold(ax, 'on');
for bd = bounds(1:end-1), xline(ax, bd + 0.5, 'Color', 'w', 'LineWidth', 1.5); end
set(ax, 'XTick', 1:numel(short), 'XTickLabel', colorLabels(short, grp, COL), 'XTickLabelRotation', 90, ...
    'YTick', 1:nT, 'YTickLabel', actNames, 'TickLabelInterpreter', 'tex', 'TickLength', [0 0]);
cb = colorbar(ax);  cb.Label.String = 'Mean z-score of members';  cb.Box = 'off';
title(ax, 'Pathway activity per replicate');
finishAxes(ax, C);  hideAxisLines(ax, C, true, true);
saveFig(fig, OUT, PDF, '27_pathway_activity_heatmap');

%% 28 Pathway activity radar ---------------------------------------------------------------------------------
nR8 = min(8, nT);  rp = 1:nR8;            % first 8 pathways of the ranked list
actR = zeros(nT, size(X,2));
for r = 1:nT, actR(r,:) = mean(zall(memb(:, topPW(r)), :), 1); end
sm = zeros(nR8, nS);
for s = 1:nS, sm(:,s) = mean(actR(rp, grp == s), 2); end
lo = floor(min(sm(:))*10)/10;  hi = ceil(max(sm(:))*10)/10;
th = linspace(0, 2*pi, nR8 + 1);
fig = newFig(89, 95);
pax = polaraxes(fig);  hold(pax, 'on');
for s = 1:nS
    plot(pax, th, [sm(:,s); sm(1,s)] - lo, 'Color', COL(s,:), 'LineWidth', 1.4);
end
pax.RLim = [0 hi - lo];
rt = linspace(0, hi - lo, 4);  pax.RTick = rt;
pax.RTickLabel = arrayfun(@(v) sprintf('%.1f', v + lo), rt, 'UniformOutput', false);
pax.ThetaTick = rad2deg(th(1:end-1));  pax.ThetaTickLabel = pwNames(topPW(rp));
pax.RAxisLocation = 360/nR8/2;  pax.GridColor = C.grid;  pax.GridAlpha = 1;
pax.FontName = 'Arial';  pax.FontSize = 6.5;  pax.RAxis.FontSize = 5.5;  pax.RColor = C.muted;  pax.ThetaColor = C.ink2;
title(pax, 'Pathway activity fingerprint', 'FontSize', 8);
legend(pax, samples, 'Box', 'off', 'Location', 'southoutside', 'Orientation', 'horizontal');
saveFig(fig, OUT, PDF, '28_pathway_activity_radar');

fprintf('28 plots saved to %s\n', OUT);

%% ======================= LOCAL FUNCTIONS ==================================
function [raw, proteins, grp, samples, pwIdx, pwNames] = simulateDataset(nP, nS, nR)
    L = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    samples = arrayfun(@(i) sprintf('Sample %s', L(i)), 1:nS, 'UniformOutput', false);
    proteins = arrayfun(@(i) sprintf('Protein_%04d', i), (1:nP)', 'UniformOutput', false);
    base = 22 + 2.2*randn(nP, 1);
    nPW = 12;  pwIdx = cell(nPW, 1);
    for k = 1:nPW, pwIdx{k} = randperm(nP, randi([60 159])); end
    pwNames = arrayfun(@(k) sprintf('Pathway %d', k), (1:nPW)', 'UniformOutput', false);
    eff = zeros(nP, nS);
    pool = unique([pwIdx{1} pwIdx{2}]);  core = pool(randperm(numel(pool), 110));
    cfc = (2*(rand(110,1) < 0.55) - 1) .* (1.9 + 0.45*randn(110,1));
    for s = 2:nS, eff(core,s) = cfc .* (1 + 0.15*randn(110,1)); end
    for s = 2:nS
        own = pwIdx{s+1};
        sp = unique([own(randperm(numel(own), min(70, numel(own)))) randperm(nP, 90)]);
        eff(sp,s) = eff(sp,s) + (2*(rand(numel(sp),1) < 0.5) - 1) .* (1.7 + 0.5*randn(numel(sp),1));
    end
    pr = pwIdx{6}(randperm(numel(pwIdx{6}), 45));
    for s = 2:min(3, nS), eff(pr,s) = eff(pr,s) - 1.8 + 0.4*randn(45,1); end
    raw = zeros(nP, nS*nR);  grp = zeros(1, nS*nR);  c = 0;
    for s = 1:nS
        for r = 1:nR
            c = c + 1;
            raw(:,c) = base + eff(:,s) + 0.35*randn(nP,1) + 0.25*randn;
            grp(c) = s;
        end
    end
    low = raw < qtile(raw(:), 0.06);
    raw((low & rand(size(raw)) < 0.65) | rand(size(raw)) < 0.012) = NaN;
end

function [p, d, t] = welchT(a, b)
    n1 = size(a,2);  n2 = size(b,2);
    v1 = var(a, 0, 2);  v2 = var(b, 0, 2);  d = mean(a,2) - mean(b,2);
    se2 = v1/n1 + v2/n2;  t = d ./ sqrt(se2);
    df = se2.^2 ./ ((v1/n1).^2/(n1-1) + (v2/n2).^2/(n2-1));
    p = betainc(df ./ (df + t.^2), df/2, 0.5);
end

function q = bhFDR(p)
    p = p(:);  n = numel(p);  [ps, o] = sort(p);
    qs = flipud(cummin(flipud(ps .* n ./ (1:n)')));
    q = zeros(n, 1);  q(o) = min(qs, 1);
end

function p = hygeUpper(k, N, K, n)
    lnC = @(a, b) gammaln(a+1) - gammaln(b+1) - gammaln(a-b+1);
    i = k:min(K, n);
    p = min(1, sum(exp(lnC(K, i) + lnC(N-K, n-i) - lnC(N, n))));
end

function v = qtile(x, q)
    x = sort(x(~isnan(x(:))));  pos = 1 + (numel(x) - 1)*q;
    lo = floor(pos);  hi = ceil(pos);  v = x(lo) + (pos - lo)*(x(hi) - x(lo));
end

function r = range2(x), r = max(x) - min(x); end

function f = kde(x, g)
    x = x(:);  h = 1.06*std(x)*numel(x)^(-1/5);
    f = zeros(size(g));
    for i = 1:numel(g), f(i) = mean(exp(-0.5*((g(i) - x)/h).^2)); end
    f = f / (h*sqrt(2*pi));
end

function D = squareDist(A)
    D = sqrt(max(sum(A.^2,2) + sum(A.^2,2)' - 2*(A*A'), 0));
end

function [Z, order] = upgma(D)
    % Average-linkage clustering on a square distance matrix.
    n = size(D,1);  D(1:n+1:end) = Inf;  active = true(1, n);  sz = ones(1, n);
    ids = 1:n;  members = num2cell(1:n);  Z = zeros(n-1, 3);
    for k = 1:n-1
        Dm = D;  Dm(~active,:) = Inf;  Dm(:,~active) = Inf;
        [m, idx] = min(Dm(:));  [i, j] = ind2sub([n n], idx);
        Z(k,:) = [ids(i) ids(j) m];
        newRow = (sz(i)*D(i,:) + sz(j)*D(j,:)) / (sz(i) + sz(j));
        D(i,:) = newRow;  D(:,i) = newRow';  D(i,i) = Inf;
        active(j) = false;  sz(i) = sz(i) + sz(j);  ids(i) = n + k;
        members{i} = [members{i} members{j}];
    end
    order = members{find(active, 1)};
end

function order = drawDendrogram(ax, Z, n, col)
    % Leaf order from the merge tree, then U-shaped links.
    kids = cell(2*n - 1, 1);
    for k = 1:n-1, kids{n+k} = Z(k, 1:2); end
    order = leafOrder(2*n - 1, kids, n);
    xpos = zeros(2*n - 1, 1);  ypos = zeros(2*n - 1, 1);
    xpos(order) = 1:n;
    for k = 1:n-1
        a = Z(k,1);  b = Z(k,2);  h = Z(k,3);
        plot(ax, [xpos(a) xpos(a) xpos(b) xpos(b)], [ypos(a) h h ypos(b)], 'Color', col, 'LineWidth', 1);
        xpos(n+k) = (xpos(a) + xpos(b)) / 2;  ypos(n+k) = h;
    end
end

function o = leafOrder(node, kids, n)
    if node <= n, o = node; return; end
    o = [leafOrder(kids{node}(1), kids, n), leafOrder(kids{node}(2), kids, n)];
end

function lab = kmeansPP(Xd, k, iters)
    n = size(Xd,1);  cen = Xd(randi(n), :);
    for c = 2:k
        d2 = min(squareDist2(Xd, cen), [], 2);
        cen(c,:) = Xd(find(cumsum(d2)/sum(d2) >= rand, 1), :);
    end
    for it = 1:iters
        [~, lab] = min(squareDist2(Xd, cen), [], 2);
        newC = cen;
        for c = 1:k, if any(lab == c), newC(c,:) = mean(Xd(lab == c, :), 1); end, end
        if max(abs(newC(:) - cen(:))) < 1e-9, break; end
        cen = newC;
    end
end

function D2 = squareDist2(A, B)
    D2 = max(sum(A.^2,2) + sum(B.^2,2)' - 2*(A*B'), 0);
end

function drawBox(ax, x, v, col, w, alpha, C)
    v = v(~isnan(v));  q1 = qtile(v, 0.25);  q2 = median(v);  q3 = qtile(v, 0.75);  iq = q3 - q1;
    lw = min(v(v >= q1 - 1.5*iq));  hw = max(v(v <= q3 + 1.5*iq));
    plot(ax, [x x], [lw q1], 'Color', C.muted, 'LineWidth', 0.6);
    plot(ax, [x x], [q3 hw], 'Color', C.muted, 'LineWidth', 0.6);
    patch(ax, x + [-w w w -w]/2, [q1 q1 q3 q3], col, 'FaceAlpha', alpha, 'EdgeColor', col);
    if alpha >= 1, mc = [1 1 1]; else, mc = C.ink; end
    plot(ax, x + [-w w]/2, [q2 q2], 'Color', mc, 'LineWidth', 1.2);
end

function drawViolin(ax, x, v, col, halfW)
    g = linspace(min(v), max(v), 200);  f = kde(v, g);  f = f / max(f) * halfW;
    patch(ax, [x + f, fliplr(x - f)], [g, fliplr(g)], col, 'EdgeColor', 'none', 'FaceAlpha', 0.85);
end

function scatterCalls(ax, xv, yv, call, C)
    scatter(ax, xv(call == 0), yv(call == 0), 3, C.ns, 'filled', 'MarkerFaceAlpha', 0.6);
    scatter(ax, xv(call == -1), yv(call == -1), 6, C.down, 'filled');
    scatter(ax, xv(call == 1), yv(call == 1), 6, C.up, 'filled');
end

function lbl = colorLabels(labels, groups, COL)
    lbl = arrayfun(@(i) sprintf('\\color[rgb]{%.3f,%.3f,%.3f}%s', COL(groups(i),:), labels{i}), ...
                   1:numel(labels), 'UniformOutput', false);
end

function sampleLegend(ax, samples, COL, loc, orient)
    h = gobjects(numel(samples), 1);
    for s = 1:numel(samples)
        h(s) = patch(ax, NaN, NaN, COL(s,:), 'EdgeColor', 'none');
    end
    legend(ax, h, samples, 'Box', 'off', 'Location', loc, 'Orientation', orient);
end

function padLimits(ax, f)
    xl = xlim(ax);  yl = ylim(ax);
    xlim(ax, xl + [-1 1]*f*diff(xl));  ylim(ax, yl + [-1 1]*f*diff(yl));
end

function fig = newFig(wmm, hmm)
    fig = figure('Units', 'centimeters', 'Position', [1 1 wmm/10 hmm/10], 'Color', 'w', 'Visible', 'off');
    set(fig, 'DefaultAxesFontName', 'Arial', 'DefaultTextFontName', 'Arial', ...
        'DefaultAxesFontSize', 6.5, 'DefaultTextFontSize', 6.5, 'DefaultLegendFontSize', 6.5, ...
        'DefaultAxesLineWidth', 0.6, 'DefaultAxesTickDir', 'out', 'DefaultAxesBox', 'off', ...
        'DefaultAxesTitleFontSizeMultiplier', 1.2, 'DefaultAxesLabelFontSizeMultiplier', 1.08);
end

function finishAxes(ax, C)
    ax.XColor = C.muted;  ax.YColor = C.muted;
    ax.XAxis.TickLabelColor = C.ink2;  ax.YAxis.TickLabelColor = C.ink2;
    ax.XLabel.Color = C.ink;  ax.YLabel.Color = C.ink;  ax.Title.Color = C.ink;
    ax.GridColor = C.grid;  ax.GridAlpha = 1;  ax.Layer = 'top';
end

function saveFig(fig, OUT, PDF, name)
    set(findall(fig, 'Type', 'Legend'), 'ItemTokenSize', [10 7]);
    exportgraphics(fig, fullfile(OUT, [name '.png']), 'Resolution', 300);
    exportgraphics(fig, PDF, 'Append', true);
    close(fig);
    fprintf('  saved %s.png\n', name);
end

function hideAxisLines(ax, C, hx, hy)
    % Hide axis lines/ticks but keep tick labels (including TeX-coloured ones).
    if hx, ax.XAxis.Color = 'none';  ax.XAxis.TickLabelColor = C.ink2; end
    if hy, ax.YAxis.Color = 'none';  ax.YAxis.TickLabelColor = C.ink2; end
    ax.XLabel.Color = C.ink;  ax.YLabel.Color = C.ink;
end

function rgb = hex2rgb(h)
    h = char(h);  rgb = [hex2dec(h(2:3)) hex2dec(h(4:5)) hex2dec(h(6:7))] / 255;
end

function cmap = makeCmap(hexes)
    stops = cell2mat(cellfun(@hex2rgb, hexes, 'UniformOutput', false)');
    cmap = interp1(linspace(0, 1, size(stops,1)), stops, linspace(0, 1, 256));
end
