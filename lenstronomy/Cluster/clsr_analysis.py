



class ClsrAnalysis(object):

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

    def plot_modeling(self,kwargs_result,multi_band_type='joint-linear'):
        model_plot = ModelPlot(self.multi_band_list, self.kwargs_model, kwargs_result, arrow_size=0.02, cmap_string="gist_heat",
                 multi_band_type=multi_band_type)
        for band_index in range(len(self.kwargs_data_joint['multi_band_list'])):
            f, axes = plt.subplots(1, 4, figsize=(16, 9))
            model_plot.data_plot(ax=axes[0], band_index=band_index)
            model_plot.model_plot(ax=axes[1], image_names=True, band_index=band_index)
            model_plot.normalized_residual_plot(ax=axes[2], v_min=-6, v_max=6, band_index=band_index)
            model_plot.source_plot(ax=axes[3],deltaPix_source=0.01, numPix=100,band_index=band_index)
            f.show()

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




