# Families

Families define the distribution and link functions for GAM models. Each family corresponds to a statistical distribution and supports specific link functions.

!!! Note
  These families correspond to thin wrappers around R families, for example from mgcv or the stats package.
  The documentation here is more concise than those in the corresponding R packages. If useful information
  is missing, please consider contributing by making a pull request!

## Continuous response
::: pymgcv.families.Gaussian
::: pymgcv.families.Gamma
::: pymgcv.families.InverseGaussian
::: pymgcv.families.Tweedie
::: pymgcv.families.Tw
::: pymgcv.families.Scat
::: pymgcv.families.MVN

## Count and proportions
::: pymgcv.families.Poisson
::: pymgcv.families.NegativeBinomial
::: pymgcv.families.ZIP
::: pymgcv.families.Betar

## Categorical and ordinal
::: pymgcv.families.Binomial
::: pymgcv.families.OCat
::: pymgcv.families.Multinom

## Location-scale
::: pymgcv.families.GauLSS
::: pymgcv.families.GammaLS
::: pymgcv.families.GevLSS
::: pymgcv.families.GumbLS
::: pymgcv.families.Shash

## Quasi-likelihood
::: pymgcv.families.Quasi
::: pymgcv.families.QuasiBinomial
::: pymgcv.families.QuasiPoisson

## Not yet implemented
::: pymgcv.families.TwLSS
::: pymgcv.families.ZipLSS
::: pymgcv.families.CNorm
::: pymgcv.families.CLog
::: pymgcv.families.CPois
::: pymgcv.families.CoxPH

## Base classes
::: pymgcv.families.AbstractFamily
::: pymgcv.families.SupportsCDF
