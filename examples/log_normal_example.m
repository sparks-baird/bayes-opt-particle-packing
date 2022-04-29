mu = 10;
sigma = 1;
pd=makedist('Lognormal','mu',mu,'sigma',sigma);
x = linspace(0,exp(mu),100);
y=pdf(pd,x);
tiledlayout(1,2)
nexttile
plot(x,y)

mu = 100;
sigma = 10;
pd2=makedist('Lognormal','mu',mu,'sigma',sigma);
x2 = linspace(0,exp(mu),100);
y2=pdf(pd2,x2);
nexttile
plot(x2,y2)

%%
mu = 10;
sigma = 1;

dist_mu = log(mu)+(-1/2).*log(1+mu.^(-2).*sigma.^2);
% dist_mu = log(mu);

% dist_sigma = log(1+mu.^(-2).*sigma.^2).^(1/2);
dist_sigma = sigma;

pd=makedist('Lognormal','mu',dist_mu,'sigma',dist_sigma);
x = random(pd,1e7,1);
cutoff = quantile(x, 0.95);
x = x(x < cutoff);
x = rescale(x);


mu = 100;
% dist_mu = log(mu);
dist_mu = log(mu)+(-1/2).*log(1+mu.^(-2).*sigma.^2);
% dist_sigma = log(1+mu.^(-2).*sigma.^2).^(1/2);

pd2=makedist('Lognormal','mu',dist_mu,'sigma',dist_sigma);
x2 = random(pd2,1e7,1);
cutoff = quantile(x2, 0.95);
x2 = x2(x2 < cutoff);
x2 = rescale(x2);

figure
tiledlayout(1,1)
nexttile
hist1 = histogram(x, 1000, 'EdgeColor', 'none', 'FaceColor',  'blue');
hold on
hist2 = histogram(x2, 1000, 'EdgeColor', 'none', 'FaceColor',  'green', 'FaceAlpha', 0.2);
% histogram([x,x2],100)
% set(gca,'YScale','log')
% nexttile
% histogram(x2,100)
% set(gca,'YScale','log')