import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.special import gammaln
from scipy.optimize import minimize

import pymc3 as pm
import theano.tensor as tt
import arviz as az

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns

import sys
from os.path import join as opj

from fg_shared import _fg_data

data_folder = opj(_fg_data, 'ncov_tcrs', 'adaptive_bio_r2')
fig_folder = opj(_fg_data, 'ncov_tcrs', 'adaptive_bio_r2', 'results', '2020-08-11')
ep_fn = opj(data_folder, 'target_v_bulk', 'FVDGVPFVV.sumcounts_tcr_counts.tcrdist_lt51.csv')
md_fn = opj(data_folder, 'target_v_bulk','joinable_metadata.csv')

d = pd.read_csv(ep_fn)
md = pd.read_csv(md_fn)

tot_counts = d[['sample_name', 'M']].drop_duplicates()

d91 = d.loc[d['i'] == 91]

d91 = pd.merge(d91.drop(['M', 'i'], axis=1), tot_counts, on='sample_name', how='right')

d91 = d91.assign(W=d91['W'].fillna(0))

d91 = pd.merge(d91, md, on='sample_name', how='left')

"""Excluding outliers for modeling exercises"""
d91 = d91.loc[d91['M'] > 1e4]
d91 = d91.loc[d91['W'] < 200]

d91 = d91.assign(sfreq=(d91['W'] + 1) / (d91['M'] + 1),
                 sfreqna=d91['W'] / d91['M'],
                 constant=1,
                 log_sfreq=np.log10((d91['W'] + 1) / (d91['M'] + 1)),
                 sex=(d91['Biological Sex'] == 'Male').astype(int))

def _jointplot(x, y, data, xbins=None, ybins=None, xscale='linear', yscale='linear', annotate=False):
    if xbins is None:
        xlim = None
    else:
        xlim = (np.min(xbins), np.max(xbins))

    if ybins is None:
        ylim = None
    else:
        ylim = (np.min(ybins), np.max(ybins))
    g = sns.JointGrid(x, y, data, xlim=xlim, ylim=ylim)
    _ = g.ax_marg_x.hist(data[x], color='b', alpha=0.6, bins=xbins)
    _ = g.ax_marg_y.hist(data[y], color='r', alpha=0.6, orientation='horizontal', bins=ybins)

    g.plot_joint(plt.scatter, color='gray', edgecolor='black', alpha=0.6)
    ax = g.ax_joint
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    g.ax_marg_x.set_xscale(xscale)
    g.ax_marg_y.set_yscale(yscale)
    if annotate:
        g = g.annotate(stats.pearsonr)
    return g

"""Make plot of frequency vs. seq depth to see if there is a relationship"""
freq_bins = np.logspace(-6, -3, 50, base=10)
tot_bins = np.logspace(4, 6.5, 50, base=10)
w_bins = np.linspace(0, 70, 50)

g = _jointplot('M', 'sfreqna', d91, xbins=tot_bins, ybins=freq_bins, xscale='log', yscale='log')

"""Plot of W vs. sfreq"""
g = _jointplot('W', 'sfreqna', d91, xbins=w_bins, ybins=freq_bins, xscale='linear', yscale='log')



tmp = d91[['W', 'days', 'sex', 'M', 'sfreq', 'log_sfreq']].dropna()

"""Negative binomial model: doesn't seem to capture the full dispersion in counts"""
model = smf.negativebinomial('W ~ sex', data=tmp, exposure=tmp['M'], loglike_method='nb2')
res = model.fit()
print(res.summary2())
tmp = tmp.assign(what=res.predict(tmp, exposure=tmp['M']),
                 resid=res.resid)
g = _jointplot('W', 'what', tmp, ybins=None, xbins=w_bins)

g = _jointplot('what', 'resid', tmp, ybins=None, xbins=None)
# g = _jointplot('W', 'resid', tmp, ybins=None, xbins=None)


"""OLS model of log-frequency"""
model = smf.ols('log_sfreq ~ sex', data=tmp)
res = model.fit()
print(res.summary2())
tmp = tmp.assign(log_sfreqhat=res.predict(tmp),
                 sfreqhat=10**res.predict(tmp),
                 resid=res.resid_pearson)
g = _jointplot('sfreq', 'sfreqhat', tmp, ybins=freq_bins, xbins=freq_bins, xscale='log', yscale='log')
g = _jointplot('log_sfreqhat', 'resid', tmp)



"""Set up a simulation of NB data to assess fit and exposure parameter"""
sz = 1000
p = 0.01
n = (np.random.randn(2*sz)*20 + 100).astype(int)

y = np.concatenate((stats.binom.rvs(n[:sz], p, size=sz),
                    stats.binom.rvs(n[sz:], p*2, size=sz)))
x = np.concatenate((np.ones(sz), np.zeros(sz)))

sim = pd.DataFrame(dict(W=y, M=n, x=x, sfreq=y/n))

model = smf.negativebinomial('W ~ x', data=sim, exposure=sim['M'])
res = model.fit()
print(res.summary2())
sim = sim.assign(what=res.predict(sim, exposure=sim['M']),
                 resid=res.resid)

g = _jointplot('W', 'x', sim)

freq_bins = np.logspace(-6, -3, 50, base=10)
tot_bins = np.logspace(4, 6.5, 50, base=10)
w_bins = np.linspace(0, 70, 50)


g = _jointplot('W', 'what', sim, ybins=None, xbins=None)

g = _jointplot('W', 'sfreq', sim, ybins=None, xbins=None)

g = _jointplot('what', 'resid', sim, ybins=None, xbins=None)
g = _jointplot('W', 'resid', sim, ybins=None, xbins=None)




with pm.Model() as model:
    alpha = pm.Exponential('alpha', 1 / sim['W'].sum())
    beta = pm.Exponential('beta', 1 / (sim['M'] - sim['W']).sum())
    obs = pm.BetaBinomial('obs', alpha, beta, sim['M'], observed=sim['W'])


with model:
    # draw 500 posterior samples
    trace = pm.sample(5000, return_inferencedata=False)

az.plot_trace(trace)

az.summary(trace, round_to=2)


with pm.Model() as betabinomial:
    
    predictor = pm.Data('predictor', sim['x'])
    trials = pm.Data('trials', sim['M'])
    
    intercept = pm.Normal('intercept', mu=np.log(0.1), sd=0.001)
    beta = pm.Normal('beta', mu=np.log(0.8), sd=0.001)
    
    p_sigma = pm.Exponential('p_sigma', 100)

    # Expected value
    p_mu = pm.math.invlogit(intercept + beta * predictor)
    p_kappa = ((p_mu * (1-p_mu))/p_sigma**2)-1 
    p_alpha = p_mu * p_kappa
    p_beta = (1-p_mu) * p_kappa
    
    # Outcome definition
    Y = pm.BetaBinomial('successes', n=trials, alpha=p_alpha, beta=p_beta, observed=sim['W'])

res = pm.find_MAP(model=model)

with model:
    trace = pm.sample(5000, return_inferencedata=False)

az.plot_trace(trace)

az.summary(trace, round_to=2)




def logp_ab(value):
    ''' prior density'''
    return tt.log(tt.pow(tt.sum(value), -5/2))

with pm.Model() as model:
    # Uninformative prior for alpha and beta
    ab = pm.HalfFlat('ab',
                     shape=2,
                     testval=np.asarray([1., 1.]))
    #pm.Potential('p(a, b)', logp_ab(ab))

    #X = pm.Deterministic('X', tt.log(ab[0]/ab[1]))
    #Z = pm.Deterministic('Z', tt.log(tt.sum(ab)))

    theta = pm.Beta('theta', alpha=ab[0], beta=ab[1], shape=tmp.shape[0])

    p = pm.Binomial('y', p=theta, observed=tmp['W'], n=tmp['M'])


res = pm.find_MAP(model=model)

with model:
    trace = pm.sample(1000, tune=2000, target_accept=0.95)

pm.traceplot(trace, var_names=['ab'])
pm.plot_posterior(trace, var_names=['ab'])

# sns.kdeplot(trace['X'], trace['Z'], shade=True, cmap='viridis')


"""Try raw hand fitting of beta binomial"""
#a, b = 5, 500000
#d91 a, b = 1.1, 170192
a, b = 2, 8e4

gby = d.loc[d['i'] == 9]
tmp = pd.merge(gby.drop(['M', 'i'], axis=1), tot_counts, on='sample_name', how='right')
tmp = tmp.assign(W=tmp['W'].fillna(0))
tmp = tmp[['W', 'M']].dropna()

p_mu = a / (a + b)
r = stats.betabinom.rvs(tmp['M'], a, b, size=tmp.shape[0])
axh = plt.subplot(111)
sns.distplot(r / tmp['M'], label='model', hist=True, kde=False, bins=freq_bins, ax=axh)
sns.distplot((tmp['W'] / tmp['M']).values, label='data', hist=True, kde=False,  bins=freq_bins, ax=axh)
axh.set_xscale('log')
plt.legend(loc=0)



"""Try fitting with scipy so that multiple clusters can be fit"""

def betabinom_nll(params, *args):
    a, b = params[0], params[1]
    k = args[0]
    n = args[1]

    logpdf = gammaln(n+1) + gammaln(k+a) + gammaln(n-k+b) + gammaln(a+b) - \
     (gammaln(k+1) + gammaln(n-k+1) + gammaln(a) + gammaln(b) + gammaln(n+a+b))

    return -np.sum(logpdf)

res = minimize(betabinom_nll, x0=[1, 1e4],
                        args=(tmp['W'], tmp['M']),
                        method='Nelder-Mead')

r = stats.betabinom.pmf(tmp['W'], tmp['M'], res['x'][0], res['x'][1])

ct = 0
for i, gby in d.groupby('i'):
    ct+=1
    tmp = pd.merge(gby.drop(['M', 'i'], axis=1), tot_counts, on='sample_name', how='right')
    tmp = tmp.assign(W=tmp['W'].fillna(0))
    tmp = tmp[['W', 'M']].dropna()
    try:
        res = minimize(betabinom_nll, x0=[2, 8e4],
                        args=(tmp['W'], tmp['M']),
                        method='Nelder-Mead')
        a, b = res['x']

        r = stats.betabinom.rvs(tmp['M'], a, b, size=tmp.shape[0])
    except ValueError:
        res = minimize(betabinom_nll, x0=[1, 1e4],
                            args=(tmp['W'], tmp['M']),
                            method='L-BFGS-B',
                            bounds=((0, np.inf), (0, np.inf)))
        a, b = res['x']

        r = stats.betabinom.rvs(tmp['M'], a, b, size=tmp.shape[0])
    
    figh = plt.figure()
    axh = plt.subplot(111)
    sns.distplot(r / tmp['M'], label='model', hist=True, kde=False, bins=freq_bins, ax=axh)
    sns.distplot((tmp['W'] / tmp['M']).values, label='data', hist=True, kde=False,  bins=freq_bins, ax=axh)
    axh.set_xscale('log')
    plt.legend(loc=0, title=i)
    if ct > 9:
        break