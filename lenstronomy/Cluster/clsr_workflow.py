from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Plots.output_plots import ModelPlot
import lenstronomy.Plots.output_plots as out_plot
import matplotlib.pyplot as plt
import numpy as np
import corner
from lenstronomy.Util.analysis_util import half_light_radius
from lenstronomy.LightModel.light_model import LightModel
import lenstronomy.Util.class_creator as class_creator
import lenstronomy.Util.util as util


class ClsrWorkflow(object):
    def __init__(self, kwargs_data_joint, kwargs_model,lens_params,source_params, kwargs_constraints, kwargs_likelihood=None):
        """
        class to  manage cluster source reconstruction.
        This class inherited the FittingSequence class in Workflow module of lenstronomy.
        :param kwargs_data_joint: keywords arguments of [data, psf, numericals] in lenstronomy convention.
        :param kwargs_model: name of model list
        :param lens_params: lens model keywords arguments [kwargs_lens_init, kwargs_lens_sigma, kwargs_fixed_lens, kwargs_lower_lens, kwargs_upper_lens]
        :param source_params: source model keywords arguments [kwargs_source_init, kwargs_source_sigma, kwargs_fixed_source, kwargs_lower_source, kwargs_upper_source]
        :param kwargs_constraints: contraints on models
        :param kwargs_likelihood: options of calculating likelihood, see more: LikelihoodModule class in Sampling module of lenstronomy.
        """
        self.kwargs_data_joint =kwargs_data_joint
        self.multi_band_list = kwargs_data_joint.get('multi_band_list', [])
        self.kwargs_model =kwargs_model
        kwargs_params = {'lens_model': lens_params, 'source_model': source_params}
        self.kwargs_params= kwargs_params
        if kwargs_constraints is None:
            kwargs_constraints ={}
        if kwargs_likelihood is None:
            kwargs_likelihood = {'source_marg': False, 'check_positive_flux': True}
        self.fitting_seq_src = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params)

    def run_fit_sequence(self,fitting_kwargs_list):
        """
        :param fitting_kwargs_list: list of [['string', {kwargs}], ..] with 'string being the specific fitting option and kwargs being the arguments passed to this option
        :return: fitting results
        """
        chain_list = self.fitting_seq_src.fit_sequence(fitting_kwargs_list)
        kwargs_result = self.fitting_seq_src.best_fit(bijective=False)
        bic_model = self.fitting_seq_src.bic
        return bic_model,chain_list, kwargs_result

    def update_fit_sequence(self,kwargs_result):
        """

        :param kwargs_result: fitting results of a specific state
        :return: go back to a specific state
        """
        self.fitting_seq_src.update_state(kwargs_result)



    def lowest_bic(self,n_max_range, n_particles,n_iterations,sigma_scale,rh, bic_model_in=[100000]):
        """

        :param n_max_range:
        :param n_particles:
        :param n_iterations:
        :param sigma_scale:
        :param img_index:
        :param Magni:
        :return:
        """
        bic_model_list = bic_model_in
        bic_in_len = len(bic_model_in)
        bic_run = True
        beta0 = rh
        chain_list_list = []
        kwargs_result_list = []
        kwargs_pso = [['PSO', {'sigma_scale': sigma_scale, 'n_particles': n_particles, 'n_iterations': n_iterations}]]
        for nmax in n_max_range:
            if nmax < 0:
                raise ValueError("nmax can not be negative!",nmax)
            else:
                if nmax == n_max_range[0]:
                    start_kwargs_shapelet = [['update_settings', {'source_remove_fixed': [  [1, ['beta'], [beta0]] ]}]]
                else:
                    start_kwargs_shapelet = []
                beta_nmax = ((nmax + 1)) ** 0.5 * beta0
                fit_kwargs_shapelet = [['update_settings',
                                        {'source_add_fixed': [[1, ['n_max'], [nmax]]],
                                        'change_source_lower_limit': [[1, ['beta'], [beta_nmax]]]
                                         }
                                         ]]
                fitting_kwargs = start_kwargs_shapelet + fit_kwargs_shapelet + kwargs_pso
            if bic_run:
                print ("nmax",nmax,"fitting_kwargs",fitting_kwargs)
                bic_model,chain_list, kwargs_result = self.run_fit_sequence(fitting_kwargs)
                chain_list_list.append(chain_list)
                kwargs_result_list.append(kwargs_result)
                if bic_model >  bic_model_list[-1]:
                    print "no neccessary to add model complexity!"
                    bic_run = False
                else:
                    print bic_model, "is the lowest BIC value"
                bic_model_list.append(bic_model)
                print "bic_model_list=", bic_model_list[bic_in_len:]
        index_bic_minima = np.where(bic_model_list == np.min(bic_model_list))[0][0] - bic_in_len
        chain_list_lowest = chain_list_list[index_bic_minima]
        kwargs_result_lowest = kwargs_result_list[index_bic_minima]
        bic_model_list_output = bic_model_list[bic_in_len:]
        return  bic_model_list_output, chain_list_lowest, kwargs_result_lowest

    def plot_chain(self, chain_list):
        """
        a fuction to plot chain_list of fitting results
        :param chain_list: chain_list of fitting results
        :return:
        """
        for i in range(len(chain_list)):
            f, axes = out_plot.plot_chain_list(chain_list, index=i)
        f.show()

    def plot_mcmc(self,chain_list_mcmc):
        sampler_type, samples_mcmc, param_mcmc, dist_mcmc = chain_list_mcmc[0]
        print("number of non-linear parameters in the MCMC process: ", len(param_mcmc))
        print("parameters in order: ", param_mcmc)
        print("number of evaluations in the MCMC process: ", np.shape(samples_mcmc)[0])
        if not samples_mcmc == []:
          corner.corner(samples_mcmc, labels=param_mcmc, show_titles=True)

    def plot_modeling(self,kwargs_result,multi_band_type='joint-linear', img_name='sys'):
        model_plot = ModelPlot(self.multi_band_list, self.kwargs_model, kwargs_result, arrow_size=0.02, cmap_string="gist_heat",
                 multi_band_type=multi_band_type)
        for band_index in range(len(self.kwargs_data_joint['multi_band_list'])):
            f, axes = plt.subplots(1, 4, figsize=(20, 9))
            model_plot.data_plot(ax=axes[0], band_index=band_index)
            model_plot.model_plot(ax=axes[1], image_names=True, band_index=band_index)
            model_plot.normalized_residual_plot(ax=axes[2], v_min=-6, v_max=6, band_index=band_index)
            model_plot.source_plot(ax=axes[3],deltaPix_source=0.01, numPix=100,band_index=band_index, scale_size =1.0)
            f.show()
            f.savefig(img_name+'source'+repr(band_index)+'.pdf')


    def source_flux_rh(self,samples_mcmc):
        """
        function to calculate flux, half light radius and their uncertainty
        :param samples_mcmc: mcmc sample
        :return:
        """
        flux_list = []
        rh_list =[]
        for i in range(len(samples_mcmc)):
            # transform the parameter position of the MCMC chain in a lenstronomy convention with keyword arguments #
            kwargs_out = self.fitting_seq_src.param_class.args2kwargs((samples_mcmc[i]))
            kwargs_source_out = kwargs_out['kwargs_source']
            imageModel = class_creator.create_im_sim(self.multi_band_list, multi_band_type='single-band',
                                                     kwargs_model=self.kwargs_model, bands_compute=[True], band_index=0)
            _,_,_,_ = imageModel.image_linear_solve(inv_bool=True, **kwargs_out)
            dp = 0.01
            x_center = kwargs_out['kwargs_source'][0]['center_x']
            y_center = kwargs_out['kwargs_source'][0]['center_y']
            x_grid_source, y_grid_source = util.make_grid(numPix=100, deltapix=dp)
            x_grid_source+= x_center
            y_grid_source+= y_center
            source_light_model = self.kwargs_model['source_light_model_list']
            lightModel = LightModel(light_model_list=source_light_model)
            source = lightModel.surface_brightness(x_grid_source,y_grid_source,kwargs_source_out)*dp ** 2
            rh = half_light_radius(source, x_grid_source, y_grid_source, x_center, y_center)
            flux = source.sum()
            flux_list.append(flux)
            rh_list.append(rh)
        return flux_list, rh_list



