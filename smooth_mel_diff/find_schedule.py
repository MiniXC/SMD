from .diffusion import DiffusionSampler

class FindSchedule():

    def __init__(self, model, diff_params, ref_cond, ref_true):
        self.model = model
        self.diff_params = diff_params
        self.ref_cond = ref_cond
        self.ref_true = ref_true

    def noise_scheduling(self, alpha_param, beta_param, ddim=False, device=None):
            """
            Start the noise scheduling process

            Parameters:
                ddim (bool): whether to use the DDIM's p_theta for noise scheduling or not
            Returns:
                ts_infer (tensor): the step indices estimated by BDDM
                a_infer (tensor):  the alphas estimated by BDDM
                b_infer (tensor):  the betas estimated by BDDM
                s_infer (tensor):  the std. deviations estimated by BDDM
            """
            max_steps = self.diff_params["N"]
            alpha = self.diff_params["alpha"]
            min_beta = self.diff_params["beta"].min()
            betas = []
            x = torch.normal(0, 1, size=self.ref_audio.shape).to(device)
            with torch.no_grad():
                b_cur = torch.ones(1, 1, 1).to(device) * beta_param
                a_cur = torch.ones(1, 1, 1).to(device) * alpha_param
                for n in range(max_steps - 1, -1, -1):
                    step = DiffusionSampler.map_noise_scale_to_time_step(a_cur.squeeze().item(), alpha)
                    if step >= 0:
                        betas.append(b_cur.squeeze().item())
                    else:
                        break
                    ts = (step * torch.ones((1, 1))).to(device)
                    e = self.model(
                        cond_mel=self.ref_cond, noisy_mel=x, step=ts
                    )
                    a_nxt = a_cur / (1 - b_cur).sqrt()
                    if ddim:
                        c1 = a_nxt / a_cur
                        c2 = -(1 - a_cur**2.).sqrt() * c1
                        x = c1 * x + c2 * e
                        c3 = (1 - a_nxt**2.).sqrt()
                        x = x + c3 * e
                    else:
                        x = x - b_cur / torch.sqrt(1 - a_cur**2.) * e
                        x = x / torch.sqrt(1 - b_cur)
                        if n > 0:
                            z = torch.normal(0, 1, size=self.ref_audio.shape).to(device)
                            x = x + torch.sqrt((1 - a_nxt**2.) / (1 - a_cur**2.) * b_cur) * z
                    a_nxt, beta_nxt = a_cur, b_cur
                    a_cur = a_nxt / (1 - beta_nxt).sqrt()
                    if a_cur > 1:
                        break
                    b_cur = self.model.schedule_net(
                        x.squeeze(1),
                        beta_next=beta_nxt.view(-1, 1)
                        delta=(1 - a_cur**2.).view(-1, 1)
                    )
                    if b_cur.squeeze().item() < min_beta:
                        break
            b_infer = torch.FloatTensor(betas[::-1]).to(device)
            a_infer = 1 - b_infer
            s_infer = b_infer + 0
            for n in range(1, len(b_infer)):
                a_infer[n] *= a_infer[n-1]
                s_infer[n] *= (1 - a_infer[n-1]) / (1 - a_infer[n])
            a_infer = torch.sqrt(a_infer)
            s_infer = torch.sqrt(s_infer)

            # Mapping noise scales to time steps
            ts_infer = []
            for n in range(len(b_infer)):
                step = DiffusionSampler.map_noise_scale_to_time_step(a_infer[n], alpha)
                if step >= 0:
                    ts_infer.append(step)
            ts_infer = torch.FloatTensor(ts_infer)
            return ts_infer, a_infer, b_infer, s_infer

        def noise_scheduling_with_params(self, alpha_param, beta_param):
            """
            Run noise scheduling for once given the (alpha_param, beta_param) pair

            Parameters:
                alpha_param (float): a hyperparameter defining the alpha_hat value at step N
                beta_param (float):  a hyperparameter defining the beta_hat value at step N
            """
            log('TRY alpha_param=%.2f, beta_param=%.2f:'%(
                alpha_param, beta_param), self.config)
            # Define the pair key
            key = '%.2f,%.2f' % (alpha_param, beta_param)
            # Set alpha_param and beta_param in self.diff_params
            # Use DDPM reverse process for noise scheduling
            ddpm_schedule = self.noise_scheduling(alpha_param, beta_param, ddim=False)
            log("\tSearched a %d-step schedule using DDPM reverse process" % (
                len(ddpm_schedule[0])), self.config)
            generated_audio, _ = self.sampling(schedule=ddpm_schedule)
            # Compute objective scores
            ddpm_score = self.assess(generated_audio)
            # Get the number of sampling steps with this schedule
            steps = len(ddpm_schedule[0])
            # Compare the performance with previous same-step schedule using the metric
            if steps not in self.steps2score:
                # Save the first schedule with this number of steps
                self.steps2score[steps] = [key, ] + ddpm_score
                self.steps2schedule[steps] = ddpm_schedule
                log('\tFound the first %d-step schedule: (PESQ, STOI) = (%.2f, %.3f)'%(
                    steps, ddpm_score[0], ddpm_score[1]), self.config)
            elif ddpm_score[0] > self.steps2score[steps][1] and ddpm_score[1] > self.steps2score[steps][2]:
                # Found a better same-step schedule achieving a higher score
                log('\tFound a better %d-step schedule: (PESQ, STOI) = (%.2f, %.3f) -> (%.2f, %.3f)'%(
                    steps, self.steps2score[steps][1], self.steps2score[steps][2],
                    ddpm_score[0], ddpm_score[1]), self.config)
                self.steps2score[steps] = [key, ] + ddpm_score
                self.steps2schedule[steps] = ddpm_schedule
            # Use DDIM reverse process for noise scheduling
            ddim_schedule = self.noise_scheduling(ddim=True)
            log("\tSearched a %d-step schedule using DDIM reverse process" % (
                len(ddim_schedule[0])), self.config)
            generated_audio, _ = self.sampling(schedule=ddim_schedule)
            # Compute objective scores
            ddim_score = self.assess(generated_audio)
            # Get the number of sampling steps with this schedule
            steps = len(ddim_schedule[0])
            # Compare the performance with previous same-step schedule using the metric
            if steps not in self.steps2score:
                # Save the first schedule with this number of steps
                self.steps2score[steps] = [key, ] + ddim_score
                self.steps2schedule[steps] = ddim_schedule
                log('\tFound the first %d-step schedule: (PESQ, STOI) = (%.2f, %.3f)'%(
                    steps, ddim_score[0], ddim_score[1]), self.config)
            elif ddim_score[0] > self.steps2score[steps][1] and ddim_score[1] > self.steps2score[steps][2]:
                # Found a better same-step schedule achieving a higher score
                log('\tFound a better %d-step schedule: (PESQ, STOI) = (%.2f, %.3f) -> (%.2f, %.3f)'%(
                    steps, self.steps2score[steps][1], self.steps2score[steps][2],
                    ddim_score[0], ddim_score[1]), self.config)
                self.steps2score[steps] = [key, ] + ddim_score
                self.steps2schedule[steps] = ddim_schedule

        def noise_scheduling_without_params(self):
            """
            Search for the best noise scheduling hyperparameters: (alpha_param, beta_param)
            """
            # Noise scheduling mode, given N
            self.reverse_process = 'BDDM'
            assert 'N' in vars(self.config).keys(), 'Error: N is undefined for BDDM!'
            self.diff_params["N"] = self.config.N
            # Init search result dictionaries
            self.steps2schedule, self.steps2score = {}, {}
            search_bins = int(self.config.bddm_search_bins)
            # Define search range of alpha_param
            alpha_last = self.diff_params["alpha"][-1].squeeze().item()
            alpha_first = self.diff_params["alpha"][0].squeeze().item()
            alpha_diff = (alpha_first - alpha_last) / (search_bins + 1)
            alpha_param_list = [alpha_last + alpha_diff * (i + 1) for i in range(search_bins)]
            # Define search range of beta_param
            beta_diff = 1. / (search_bins + 1)
            beta_param_list = [beta_diff * (i + 1) for i in range(search_bins)]
            # Search for beta_param and alpha_param, take O(search_bins^2)
            for beta_param in beta_param_list:
                for alpha_param in alpha_param_list:
                    if alpha_param > (1 - beta_param) ** 0.5:
                        # Invalid range
                        continue
                    # Update the scores and noise schedules with (alpha_param, beta_param)
                    self.noise_scheduling_with_params(alpha_param, beta_param)
            # Lastly, repeat the random starting point (x_hat_N) and choose the best schedule
            noise_schedule_dir = os.path.join(self.exp_dir, 'noise_schedules')
            os.makedirs(noise_schedule_dir, exist_ok=True)
            steps_list = list(self.steps2score.keys())
            for steps in steps_list:
                log("-"*80, self.config)
                log("Select the best out of %d x_hat_N ~ N(0,I) for %d steps:"%(
                    self.config.noise_scheduling_attempts, steps), self.config)
                # Get current best pair
                key = self.steps2score[steps][0]
                # Get back the best (alpha_param, beta_param) pair for a given steps
                alpha_param, beta_param = list(map(float, key.split(',')))
                # Repeat K times for a given number of steps
                for _ in range(self.config.noise_scheduling_attempts):
                    # Random +/- 5%
                    _alpha_param = alpha_param * (0.95 + np.random.rand() * 0.1)
                    _beta_param = beta_param * (0.95 + np.random.rand() * 0.1)
                    # Update the scores and noise schedules with (alpha_param, beta_param)
                    self.noise_scheduling_with_params(_alpha_param, _beta_param)
            # Save the best searched noise schedule ({N}steps_{key}_{metric}{best_score}.ns)
            for steps in sorted(self.steps2score.keys(), reverse=True):
                filepath = os.path.join(noise_schedule_dir, '%dsteps_PESQ%.2f_STOI%.3f.ns'%(
                    steps, self.steps2score[steps][1], self.steps2score[steps][2]))
                torch.save(self.steps2schedule[steps], filepath)
                log("Saved searched schedule: %s" % filepath, self.config)