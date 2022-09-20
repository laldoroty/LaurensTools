from cgi import test
from copy import copy
import numpy as np
from astropy.cosmology import FlatLambdaCDM

def cosmo_loocv(model=None,*args):
    """
    LNA 20220914

    Leave-one-out-cross-validation for the chisq minimization cosmology
    models in this directory. Must use args in the same order as specified
    in the comments, which are the same as the order in the actual functions.

    Arguments:
    model -- Specify the chisq minimization model. Can be 'cmagic', 'salt', or 'tripp'.
    args -- the args you'd put in to cmagicmodel.cmagicmodel(), saltmodel.saltmodel(), or trippmodel.trippmodel().
    """

    # First, set up based on model.
    # Get the chisq minimization routine,
    # and define the distance modulus function.
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    acceptable_models = ['cmagic','salt','tripp']
    if model not in acceptable_models:
        raise ValueError(f'Acceptable models for model argument are {acceptable_models}.')
    elif model == 'cmagic':
        """
        Use args: bbv,ebbv,bmax,ebmax,dm15,edm15,slope,eslope,zcmb,model='He2018'
        you can choose between 'He2018' and 'Aldoroty2022' for model arg.
        """
        from .cmagicmodel import cmagicmodel as chisq_func
        bbv,ebbv,bmax,ebmax,dm15,edm15,slope,eslope,zcmb,model = args
        args = [bbv,ebbv,bmax,ebmax,dm15,edm15,slope,eslope,zcmb]

        def test_dm(train_pars,test,*dat):
            M,delta,b2 = train_pars
            bbv,_,bmax,_,dm15,_,slope,_,_ = dat
            t_bbv,_,t_bmax,_,t_dm15,_,t_slope,_,t_zcmb = test
            if model == 'He2018':
                mu = t_bbv - M - delta*(t_dm15 - np.mean(dm15)) - (b2 - t_slope)*((t_bmax-t_bbv)/t_slope + 1.2*(1/t_slope - np.mean(1/slope)))
            elif model == 'Aldoroty2022':
                mu = t_bbv - M - delta*(t_dm15 - np.mean(dm15)) - (b2 - t_slope)*((t_bmax-t_bbv)/t_slope - np.mean((bmax-bbv)/slope))
            mu_expected = cosmo.distmod(t_zcmb).value
            resid = mu - mu_expected
            return mu, resid, t_zcmb


    elif model == 'salt':
        """
        Use args: bmax,ebmax,x1,ex1,c,ec,zcmb
        """
        from .saltmodel import saltmodel as chisq_func
        def test_dm(train_pars,test,*dat):
            print('SALT test_dm')
            print(train_pars)
            print(test)
            print(dat)
            M,a,b = train_pars
            bmax,_,x1,_,c,_,_ = dat
            t_bmax,_,t_x1,_,t_c,_,t_zcmb = test
            mu = t_bmax - M + a*(t_x1) - b*(t_c)
            mu_expected = cosmo.distmod(t_zcmb).value
            resid = mu - mu_expected
            return mu, resid, t_zcmb

    elif model == 'tripp':
        """
        Use args: bmax,ebmax,c,ec,dm15,edm15,zcmb
        """
        from .trippmodel import trippmodel as chisq_func
        def test_dm(train_pars,test,*dat):
            M,a,d = train_pars
            bmax,_,c,_,dm15,_,_ = dat
            t_bmax,_,t_c,_,t_dm15,_,t_zcmb = test
            mu = t_bmax - M - a*(t_c-np.mean(c)) - d*(t_dm15-np.mean(dm15))
            mu_expected = cosmo.distmod(t_zcmb).value
            resid = mu - mu_expected
            return mu, resid, t_zcmb

    # Now, use all this!
    N = len(args[0])
    args_copy = np.array(copy(args))
    test_mus = []
    test_zcmbs = []
    test_resids = []
    for i in range(N):
        train_args = np.delete(args_copy,i,axis=1)
        test_args = args_copy[:,i]
        train_params = chisq_func(*train_args)[0] # [0] is the final param estimate, theta_fin
        test_mu, test_resid, test_zcmb = test_dm(train_params, test_args, *train_args)
        test_resids.append(test_resid)
        test_mus.append(test_mu)
        test_zcmbs.append(test_zcmb)

    return test_resids, test_mus, test_zcmbs