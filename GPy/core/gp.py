# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import sys
from .. import kern
from GPy.kern._src.rbf import RBF
from GPy.likelihoods.gaussian import Gaussian
from model import Model
from parameterization import ObsAr
from .. import likelihoods
from ..inference.latent_function_inference import exact_gaussian_inference, expectation_propagation
from parameterization.variational import VariationalPosterior

import logging
from GPy.util.normalizer import MeanNorm
logger = logging.getLogger("GP")

class GP(Model):
    """
    General purpose Gaussian process model

    :param X: input observations
    :param Y: output observations
    :param kernel: a GPy kernel, defaults to rbf+white
    :param likelihood: a GPy likelihood
    :param inference_method: The :class:`~GPy.inference.latent_function_inference.LatentFunctionInference` inference method to use for this GP
    :rtype: model object
    :param Norm normalizer:
        normalize the outputs Y.
        Prediction will be un-normalized using this normalizer.
        If normalizer is None, we will normalize using MeanNorm.
        If normalizer is False, no normalization will be done.

    .. Note:: Multiple independent outputs are allowed using columns of Y


    """
    def __init__(self, X, Y, kernel, likelihood, inference_method=None, name='gp', Y_metadata=None, normalizer=False):
        super(GP, self).__init__(name)

        assert X.ndim == 2
        if isinstance(X, (ObsAr, VariationalPosterior)):
            self.X = X.copy()
        else: self.X = ObsAr(X)

        self.num_data, self.input_dim = self.X.shape

        assert Y.ndim == 2
        logger.info("initializing Y")

        if normalizer is True:
            self.normalizer = MeanNorm()
        elif normalizer is False:
            self.normalizer = None
        else:
            self.normalizer = normalizer

        if self.normalizer is not None:
            self.normalizer.scale_by(Y)
            self.Y_normalized = ObsAr(self.normalizer.normalize(Y))
            self.Y = Y
        else:
            self.Y = ObsAr(Y)
            self.Y_normalized = self.Y

        assert Y.shape[0] == self.num_data
        _, self.output_dim = self.Y.shape

        #TODO: check the type of this is okay?
        self.Y_metadata = Y_metadata

        assert isinstance(kernel, kern.Kern)
        #assert self.input_dim == kernel.input_dim
        self.kern = kernel

        assert isinstance(likelihood, likelihoods.Likelihood)
        self.likelihood = likelihood

        #find a sensible inference method
        logger.info("initializing inference method")
        if inference_method is None:
            if isinstance(likelihood, likelihoods.Gaussian) or isinstance(likelihood, likelihoods.MixedNoise):
                inference_method = exact_gaussian_inference.ExactGaussianInference()
            else:
                inference_method = expectation_propagation.EP()
                print "defaulting to ", inference_method, "for latent function inference"
        self.inference_method = inference_method

        logger.info("adding kernel and likelihood as parameters")
        self.link_parameter(self.kern)
        self.link_parameter(self.likelihood)

    def set_XY(self, X=None, Y=None):
        """
        Set the input / output data of the model
        This is useful if we wish to change our existing data but maintain the same model

        :param X: input observations
        :type X: np.ndarray
        :param Y: output observations
        :type Y: np.ndarray
        """
        self.update_model(False)
        if Y is not None:
            if self.normalizer is not None:
                self.normalizer.scale_by(Y)
                self.Y_normalized = ObsAr(self.normalizer.normalize(Y))
                self.Y = Y
            else:
                self.Y = ObsAr(Y)
                self.Y_normalized = self.Y
        if X is not None:
            if self.X in self.parameters:
                # LVM models
                if isinstance(self.X, VariationalPosterior):
                    assert isinstance(X, type(self.X)), "The given X must have the same type as the X in the model!"
                    self.unlink_parameter(self.X)
                    self.X = X
                    self.link_parameters(self.X)
                else:
                    self.unlink_parameter(self.X)
                    from ..core import Param
                    self.X = Param('latent mean',X)
                    self.link_parameters(self.X)
            else:
                self.X = ObsAr(X)
        self.update_model(True)
        self._trigger_params_changed()

    def set_X(self,X):
        """
        Set the input data of the model

        :param X: input observations
        :type X: np.ndarray
        """
        self.set_XY(X=X)

    def set_Y(self,Y):
        """
        Set the output data of the model

        :param X: output observations
        :type X: np.ndarray
        """
        self.set_XY(Y=Y)

    def parameters_changed(self):
        """
        Method that is called upon any changes to :class:`~GPy.core.parameterization.param.Param` variables within the model.
        In particular in the GP class this method reperforms inference, recalculating the posterior and log marginal likelihood and gradients of the model

        .. warning::
            This method is not designed to be called manually, the framework is set up to automatically call this method upon changes to parameters, if you call
            this method yourself, there may be unexpected consequences.
        """
        self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern, self.X, self.likelihood, self.Y_normalized, self.Y_metadata)
        self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])
        self.kern.update_gradients_full(self.grad_dict['dL_dK'], self.X)

    def log_likelihood(self):
        """
        The log marginal likelihood of the model, :math:`p(\mathbf{y})`, this is the objective function of the model being optimised
        """
        return self._log_marginal_likelihood

    def _raw_predict(self, _Xnew, full_cov=False, kern=None):
        """
        For making predictions, does not account for normalization or likelihood

        full_cov is a boolean which defines whether the full covariance matrix
        of the prediction is computed. If full_cov is False (default), only the
        diagonal of the covariance is returned.

        .. math::
            p(f*|X*, X, Y) = \int^{\inf}_{\inf} p(f*|f,X*)p(f|X,Y) df
                        = N(f*| K_{x*x}(K_{xx} + \Sigma)^{-1}Y, K_{x*x*} - K_{xx*}(K_{xx} + \Sigma)^{-1}K_{xx*}
            \Sigma := \texttt{Likelihood.variance / Approximate likelihood covariance}
        """
        if kern is None:
            kern = self.kern

        Kx = kern.K(_Xnew, self.X).T
        WiKx = np.dot(self.posterior.woodbury_inv, Kx)
        mu = np.dot(Kx.T, self.posterior.woodbury_vector)
        if full_cov:
            Kxx = kern.K(_Xnew)
            var = Kxx - np.dot(Kx.T, WiKx)
        else:
            Kxx = kern.Kdiag(_Xnew)
            var = Kxx - np.sum(WiKx*Kx, 0)
            var = var.reshape(-1, 1)

        #force mu to be a column vector
        if len(mu.shape)==1: mu = mu[:,None]
        return mu, var

    def predict_uncertain(self, Xnew_mu, Xnew_covar):
        """
        Predict the functions' mean and variance at the new points distributed by Xnew, where Xnew is assumed
        to be normal distributed with mean Xnew_mu and variance Xnew_var.
        This function only works for an RBF kernel.

        :param Xnew_mu: The mean of the points at which to make a prediction
        :type Xnew_mu: np.ndarray (Nnew x self.input_dim)
        :param Xnew_covar: The covariance w.r.t. the input dimensions of the points at which to make a prediction
        :type Xnew_covar: np.ndarray (Nnew x self.input_dim x self.input_dim)
        :returns: (mean, covar):
            mean: posterior mean, a Numpy array, Nnew x self.output_dim
            covar: posterior covariance, a Numpy array, Nnew x self.output_dim x self.output_dim

           Note that the covariance is calculated w.r.t. the output dimensions for each sample
           and not w.r.t. to the samples, as it is common with GPs.
        """

        if not isinstance(self.kern, RBF):
            raise ValueError("predict_from_unsure_input only works with RBF kernel")
        if not isinstance(self.likelihood, Gaussian):
            raise ValueError("predict_from_unsure_input only works with Gaussian likelihood")
        if Xnew_mu.shape[0] != Xnew_covar.shape[0]:
            raise ValueError("Xnew_mu and Xnew_covar must have same number of samples")
        if (Xnew_mu.shape[1] != self.input_dim or Xnew_covar.shape[1] != self.input_dim or
                Xnew_covar.shape[2] != self.input_dim):
            raise ValueError("Wrong number of input dimensions provided")

        #######################################################################################
        # inputs:
        n_smpls = Xnew_mu.shape[0]
        # mu[new_smpl, feature] = Xnew_mu
        mu = Xnew_mu
        # Sigma[new_smpl, feature, feature] = Xnew_covar
        Sigma = Xnew_covar

        # kernel parameters:
        # ls[feature]
        ls = 1. / self.kern.lengthscale ** 2
        if ls.shape[0] == 1:
            ls = ls * np.ones(self.input_dim)
        # LambdaInv[feature, feature]
        LambdaInv = np.diag(ls)
        # Lambda[feature, feature]
        Lambda = np.diag(1. / ls)

        # alpha2 (scalar)
        alpha2 = np.asarray(self.kern.variance)

        # SigmaE2 (scalar)
        SigmaE2 = np.asarray(self.likelihood.variance)

        # print "alpha2=", alpha2

        # training data:
        # self.X[X_smpl, feature]
        # self.Y[X_smpl, out_dim]

        # dxx[X_smpl, X_smpl, feature]
        dxx = self.X[:,np.newaxis,:] - self.X[np.newaxis,:,:]

        # DxxLambdaInvDxx[X_smpl, X_smpl]
        DxxLambdaInvDxx = np.sum(ls[np.newaxis,np.newaxis,:] * dxx**2, axis=2)

        # Kxx[X_smpl, X_smpl]
        #Kxx = alpha2 * np.exp(-0.5 * DxxLambdaInvDxx)

        # KxxPlusSigmaE[X_smpl, X_smpl]
        #KxxPlusSigmaE = Kxx + SigmaE2 * np.identity(Kxx.shape[0])

        # Beta[X_smpl, X_smpl] = woodbury_inv
        #Beta = np.linalg.inv(KxxPlusSigmaE)
        Beta = self.posterior.woodbury_inv

        # beta[X_smpl, out_dim] = woodbury_vector
        #beta = np.dot(Beta, self.Y)
        beta = self.posterior.woodbury_vector
        #######################################################################################

        #######################################################################################
        # calculate predictive mean, eq. (9)

        # dxmu[new_smpl, X_smpl, feature]
        dxmu = self.X[np.newaxis,:,:] - mu[:,np.newaxis,:]

        # SigmaPlusLambda[new_xmpl, feature, feature]
        SigmaPlusLambda = Sigma + Lambda[np.newaxis,:,:]

        # SigmaPlusLambdaInv[new_xmpl, feature, feature]
        SigmaPlusLambdaInv = np.zeros_like(SigmaPlusLambda)
        for s in range(n_smpls):
            SigmaPlusLambdaInv[s,:,:] = np.linalg.inv(SigmaPlusLambda[s,:,:])

        # DSigmaPlusLambdaInvD[new_smpl, X_smpl]
        DSigmaPlusLambdaInvD = np.einsum("sif,sfg,sig->si", dxmu, SigmaPlusLambdaInv, dxmu)

        # SigmaLambdaInvPlusId[new_smpl, feature, feature]
        SigmaLambdaInvPlusId = np.einsum("sik,kj->sij", Sigma, LambdaInv) + np.identity(self.input_dim)[np.newaxis,:,:]

        # SigmaLambdaInvPlusIdDet[new_smpl]
        SigmaLambdaInvPlusIdDet = np.zeros((n_smpls,))
        for s in range(n_smpls):
            SigmaLambdaInvPlusIdDet[s] = np.linalg.det(SigmaLambdaInvPlusId[s,:,:])

        # l[new_smpl, X_smpl]
        l = alpha2 * (SigmaLambdaInvPlusIdDet**(-0.5))[:,np.newaxis] * np.exp(-0.5 * DSigmaPlusLambdaInvD)

        # p_mean[new_smpl, out_dim]
        p_mean = np.dot(l, beta)
        #######################################################################################

        #######################################################################################
        # calculate predictive covariance of output dimensions

        # k* = ks[new_smpl, X_smpl]
        ks = alpha2 * np.exp(-0.5 * np.sum(ls[np.newaxis,np.newaxis,:] * dxmu**2, axis=2))

        # LambdaInvSigma[new_smpl, feature, feature]
        LambdaInvSigma = np.einsum("fg,sgh->sfh", LambdaInv, Sigma)

        # TwoLambdaInvSigmaPlusId[new_smpl, feature, feature]
        TwoLambdaInvSigmaPlusId = 2 * LambdaInvSigma + np.identity(self.input_dim)[np.newaxis,:,:]

        # TwoLambdaInvSigmaPlusIdDet[new_smpl]
        TwoLambdaInvSigmaPlusIdDet = np.zeros((n_smpls,))
        for s in range(n_smpls):
            TwoLambdaInvSigmaPlusIdDet[s] = np.linalg.det(TwoLambdaInvSigmaPlusId[s,:,:])

        # z[X_smpl, X_smpl, feature]
        z = 0.5 * (self.X[:,np.newaxis,:] + self.X[np.newaxis,:,:])

        # gamma[new_smpl, X_smpl, X_smpl, feature]
        gamma = z[np.newaxis,:,:,:] - mu[:,np.newaxis,np.newaxis,:]

        # HalfLambdaPlusSigma[new_smpl, feature, feature]
        HalfLambdaPlusSigma = 0.5 * Lambda[np.newaxis,:,:] + Sigma

        # HalfLambdaPlusSigmaInv[new_smpl, feature, feature]
        HalfLambdaPlusSigmaInv = np.zeros_like(HalfLambdaPlusSigma)
        for s in range(n_smpls):
            HalfLambdaPlusSigmaInv[s,:,:] = np.linalg.inv(HalfLambdaPlusSigma[s,:,:])

        # GammaHalfLambdaPlusSigmaInvGamma[new_smpl, X_smpl, X_smpl]
        GammaHalfLambdaPlusSigmaInvGamma = np.einsum("sijf,sfg,sijg->sij", gamma, HalfLambdaPlusSigmaInv, gamma)

        # L[new_smpl, X_smpl, X_smpl], eq. (12)
        L = (alpha2**2 * TwoLambdaInvSigmaPlusIdDet[:,np.newaxis,np.newaxis]**(-0.5) *
             np.exp(-0.25 * DxxLambdaInvDxx[np.newaxis,:,:]) *
             np.exp(-0.5 * GammaHalfLambdaPlusSigmaInvGamma))

        # Eh[new_smpl, out_dim, out_dim]
        Eh = np.einsum("xo,mxy,yp->mop", beta, L, beta)

        # BetaL[new_smpl, X_smpl, X_smpl]
        BetaL = np.einsum("xy,syz->sxz", Beta, L)

        # TrBetaL[new_smpl]
        TrBetaL = np.einsum("sxx->s", BetaL)

        # Alpha2PlusTrBetaL[new_smpl, out_dim, out_dim]
        Alpha2PlusTrBetaL = np.identity(self.output_dim)[np.newaxis,:,:] * (alpha2 - TrBetaL[:,np.newaxis,np.newaxis] + SigmaE2)

        # p_cov[new_smpl, out_dim, out_dim]
        p_cov = Eh - p_mean[:,:,np.newaxis] * p_mean[:,np.newaxis,:] + Alpha2PlusTrBetaL

        #######################################################################################

        return p_mean, p_cov

    def predict(self, Xnew, full_cov=False, Y_metadata=None, kern=None):
        """
        Predict the function(s) at the new point(s) Xnew.

        :param Xnew: The points at which to make a prediction
        :type Xnew: np.ndarray (Nnew x self.input_dim)
        :param full_cov: whether to return the full covariance matrix, or just
                         the diagonal
        :type full_cov: bool
        :param Y_metadata: metadata about the predicting point to pass to the likelihood
        :param kern: The kernel to use for prediction (defaults to the model
                     kern). this is useful for examining e.g. subprocesses.
        :returns: (mean, var, lower_upper):
            mean: posterior mean, a Numpy array, Nnew x self.input_dim
            var: posterior variance, a Numpy array, Nnew x 1 if full_cov=False, Nnew x Nnew otherwise
            lower_upper: lower and upper boundaries of the 95% confidence intervals, Numpy arrays,  Nnew x self.input_dim

           If full_cov and self.input_dim > 1, the return shape of var is Nnew x Nnew x self.input_dim. If self.input_dim == 1, the return shape is Nnew x Nnew.
           This is to allow for different normalizations of the output dimensions.
        """
        #predict the latent function values
        mu, var = self._raw_predict(Xnew, full_cov=full_cov, kern=kern)
        if self.normalizer is not None:
            mu, var = self.normalizer.inverse_mean(mu), self.normalizer.inverse_variance(var)

        # now push through likelihood
        mean, var = self.likelihood.predictive_values(mu, var, full_cov, Y_metadata)
        return mean, var

    def predict_quantiles(self, X, quantiles=(2.5, 97.5), Y_metadata=None):
        """
        Get the predictive quantiles around the prediction at X

        :param X: The points at which to make a prediction
        :type X: np.ndarray (Xnew x self.input_dim)
        :param quantiles: tuple of quantiles, default is (2.5, 97.5) which is the 95% interval
        :type quantiles: tuple
        :returns: list of quantiles for each X and predictive quantiles for interval combination
        :rtype: [np.ndarray (Xnew x self.input_dim), np.ndarray (Xnew x self.input_dim)]
        """
        m, v = self._raw_predict(X,  full_cov=False)
        if self.normalizer is not None:
            m, v = self.normalizer.inverse_mean(m), self.normalizer.inverse_variance(v)
        return self.likelihood.predictive_quantiles(m, v, quantiles, Y_metadata)

    def predictive_gradients(self, Xnew):
        """
        Compute the derivatives of the latent function with respect to X*

        Given a set of points at which to predict X* (size [N*,Q]), compute the
        derivatives of the mean and variance. Resulting arrays are sized:
         dmu_dX* -- [N*, Q ,D], where D is the number of output in this GP (usually one).

         dv_dX*  -- [N*, Q],    (since all outputs have the same variance)
        :param X: The points at which to get the predictive gradients
        :type X: np.ndarray (Xnew x self.input_dim)
        :returns: dmu_dX, dv_dX
        :rtype: [np.ndarray (N*, Q ,D), np.ndarray (N*,Q) ]

        """
        dmu_dX = np.empty((Xnew.shape[0],Xnew.shape[1],self.output_dim))
        for i in range(self.output_dim):
            dmu_dX[:,:,i] = self.kern.gradients_X(self.posterior.woodbury_vector[:,i:i+1].T, Xnew, self.X)

        # gradients wrt the diagonal part k_{xx}
        dv_dX = self.kern.gradients_X(np.eye(Xnew.shape[0]), Xnew)
        #grads wrt 'Schur' part K_{xf}K_{ff}^{-1}K_{fx}
        alpha = -2.*np.dot(self.kern.K(Xnew, self.X),self.posterior.woodbury_inv)
        dv_dX += self.kern.gradients_X(alpha, Xnew, self.X)
        return dmu_dX, dv_dX


    def posterior_samples_f(self,X,size=10, full_cov=True):
        """
        Samples the posterior GP at the points X.

        :param X: The points at which to take the samples.
        :type X: np.ndarray (Nnew x self.input_dim)
        :param size: the number of a posteriori samples.
        :type size: int.
        :param full_cov: whether to return the full covariance matrix, or just the diagonal.
        :type full_cov: bool.
        :returns: Ysim: set of simulations
        :rtype: np.ndarray (N x samples)
        """
        m, v = self._raw_predict(X,  full_cov=full_cov)
        if self.normalizer is not None:
            m, v = self.normalizer.inverse_mean(m), self.normalizer.inverse_variance(v)
        v = v.reshape(m.size,-1) if len(v.shape)==3 else v
        if not full_cov:
            Ysim = np.random.multivariate_normal(m.flatten(), np.diag(v.flatten()), size).T
        else:
            Ysim = np.random.multivariate_normal(m.flatten(), v, size).T

        return Ysim

    def posterior_samples(self, X, size=10, full_cov=False, Y_metadata=None):
        """
        Samples the posterior GP at the points X.

        :param X: the points at which to take the samples.
        :type X: np.ndarray (Nnew x self.input_dim.)
        :param size: the number of a posteriori samples.
        :type size: int.
        :param full_cov: whether to return the full covariance matrix, or just the diagonal.
        :type full_cov: bool.
        :param noise_model: for mixed noise likelihood, the noise model to use in the samples.
        :type noise_model: integer.
        :returns: Ysim: set of simulations, a Numpy array (N x samples).
        """
        Ysim = self.posterior_samples_f(X, size, full_cov=full_cov)
        Ysim = self.likelihood.samples(Ysim, Y_metadata)

        return Ysim

    def plot_f(self, plot_limits=None, which_data_rows='all',
        which_data_ycols='all', fixed_inputs=[],
        levels=20, samples=0, fignum=None, ax=None, resolution=None,
        plot_raw=True,
        linecol=None,fillcol=None, Y_metadata=None, data_symbol='kx'):
        """
        Plot the GP's view of the world, where the data is normalized and before applying a likelihood.
        This is a call to plot with plot_raw=True.
        Data will not be plotted in this, as the GP's view of the world
        may live in another space, or units then the data.

        Can plot only part of the data and part of the posterior functions
        using which_data_rowsm which_data_ycols.

        :param plot_limits: The limits of the plot. If 1D [xmin,xmax], if 2D [[xmin,ymin],[xmax,ymax]]. Defaluts to data limits
        :type plot_limits: np.array
        :param which_data_rows: which of the training data to plot (default all)
        :type which_data_rows: 'all' or a slice object to slice model.X, model.Y
        :param which_data_ycols: when the data has several columns (independant outputs), only plot these
        :type which_data_ycols: 'all' or a list of integers
        :param fixed_inputs: a list of tuple [(i,v), (i,v)...], specifying that input index i should be set to value v.
        :type fixed_inputs: a list of tuples
        :param resolution: the number of intervals to sample the GP on. Defaults to 200 in 1D and 50 (a 50x50 grid) in 2D
        :type resolution: int
        :param levels: number of levels to plot in a contour plot.
        :param levels: for 2D plotting, the number of contour levels to use is ax is None, create a new figure
        :type levels: int
        :param samples: the number of a posteriori samples to plot
        :type samples: int
        :param fignum: figure to plot on.
        :type fignum: figure number
        :param ax: axes to plot on.
        :type ax: axes handle
        :param linecol: color of line to plot [Tango.colorsHex['darkBlue']]
        :type linecol: color either as Tango.colorsHex object or character ('r' is red, 'g' is green) as is standard in matplotlib
        :param fillcol: color of fill [Tango.colorsHex['lightBlue']]
        :type fillcol: color either as Tango.colorsHex object or character ('r' is red, 'g' is green) as is standard in matplotlib
        :param Y_metadata: additional data associated with Y which may be needed
        :type Y_metadata: dict
        :param data_symbol: symbol as used matplotlib, by default this is a black cross ('kx')
        :type data_symbol: color either as Tango.colorsHex object or character ('r' is red, 'g' is green) alongside marker type, as is standard in matplotlib.
        """
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import models_plots
        kw = {}
        if linecol is not None:
            kw['linecol'] = linecol
        if fillcol is not None:
            kw['fillcol'] = fillcol
        return models_plots.plot_fit(self, plot_limits, which_data_rows,
                                     which_data_ycols, fixed_inputs,
                                     levels, samples, fignum, ax, resolution,
                                     plot_raw=plot_raw, Y_metadata=Y_metadata,
                                     data_symbol=data_symbol, **kw)

    def plot(self, plot_limits=None, which_data_rows='all',
        which_data_ycols='all', fixed_inputs=[],
        levels=20, samples=0, fignum=None, ax=None, resolution=None,
        plot_raw=False,
        linecol=None,fillcol=None, Y_metadata=None, data_symbol='kx'):
        """
        Plot the posterior of the GP.
          - In one dimension, the function is plotted with a shaded region identifying two standard deviations.
          - In two dimsensions, a contour-plot shows the mean predicted function
          - In higher dimensions, use fixed_inputs to plot the GP  with some of the inputs fixed.

        Can plot only part of the data and part of the posterior functions
        using which_data_rowsm which_data_ycols.

        :param plot_limits: The limits of the plot. If 1D [xmin,xmax], if 2D [[xmin,ymin],[xmax,ymax]]. Defaluts to data limits
        :type plot_limits: np.array
        :param which_data_rows: which of the training data to plot (default all)
        :type which_data_rows: 'all' or a slice object to slice model.X, model.Y
        :param which_data_ycols: when the data has several columns (independant outputs), only plot these
        :type which_data_ycols: 'all' or a list of integers
        :param fixed_inputs: a list of tuple [(i,v), (i,v)...], specifying that input index i should be set to value v.
        :type fixed_inputs: a list of tuples
        :param resolution: the number of intervals to sample the GP on. Defaults to 200 in 1D and 50 (a 50x50 grid) in 2D
        :type resolution: int
        :param levels: number of levels to plot in a contour plot.
        :param levels: for 2D plotting, the number of contour levels to use is ax is None, create a new figure
        :type levels: int
        :param samples: the number of a posteriori samples to plot
        :type samples: int
        :param fignum: figure to plot on.
        :type fignum: figure number
        :param ax: axes to plot on.
        :type ax: axes handle
        :param linecol: color of line to plot [Tango.colorsHex['darkBlue']]
        :type linecol: color either as Tango.colorsHex object or character ('r' is red, 'g' is green) as is standard in matplotlib
        :param fillcol: color of fill [Tango.colorsHex['lightBlue']]
        :type fillcol: color either as Tango.colorsHex object or character ('r' is red, 'g' is green) as is standard in matplotlib
        :param Y_metadata: additional data associated with Y which may be needed
        :type Y_metadata: dict
        :param data_symbol: symbol as used matplotlib, by default this is a black cross ('kx')
        :type data_symbol: color either as Tango.colorsHex object or character ('r' is red, 'g' is green) alongside marker type, as is standard in matplotlib.
        """
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import models_plots
        kw = {}
        if linecol is not None:
            kw['linecol'] = linecol
        if fillcol is not None:
            kw['fillcol'] = fillcol
        return models_plots.plot_fit(self, plot_limits, which_data_rows,
                                     which_data_ycols, fixed_inputs,
                                     levels, samples, fignum, ax, resolution,
                                     plot_raw=plot_raw, Y_metadata=Y_metadata,
                                     data_symbol=data_symbol, **kw)

    def input_sensitivity(self, summarize=True):
        """
        Returns the sensitivity for each dimension of this model
        """
        return self.kern.input_sensitivity(summarize=summarize)

    def optimize(self, optimizer=None, start=None, **kwargs):
        """
        Optimize the model using self.log_likelihood and self.log_likelihood_gradient, as well as self.priors.
        kwargs are passed to the optimizer. They can be:

        :param max_f_eval: maximum number of function evaluations
        :type max_f_eval: int
        :messages: whether to display during optimisation
        :type messages: bool
        :param optimizer: which optimizer to use (defaults to self.preferred optimizer), a range of optimisers can be found in :module:`~GPy.inference.optimization`, they include 'scg', 'lbfgs', 'tnc'.
        :type optimizer: string
        """
        self.inference_method.on_optimization_start()
        try:
            super(GP, self).optimize(optimizer, start, **kwargs)
        except KeyboardInterrupt:
            print "KeyboardInterrupt caught, calling on_optimization_end() to round things up"
            self.inference_method.on_optimization_end()
            raise

    def infer_newX(self, Y_new, optimize=True, ):
        """
        Infer the distribution of X for the new observed data *Y_new*.

        :param Y_new: the new observed data for inference
        :type Y_new: numpy.ndarray
        :param optimize: whether to optimize the location of new X (True by default)
        :type optimize: boolean
        :return: a tuple containing the posterior estimation of X and the model that optimize X
        :rtype: (:class:`~GPy.core.parameterization.variational.VariationalPosterior` or numpy.ndarray, :class:`~GPy.core.model.Model`)
        """
        from ..inference.latent_function_inference.inferenceX import infer_newX
        return infer_newX(self, Y_new, optimize=optimize)
