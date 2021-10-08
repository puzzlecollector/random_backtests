class Agent: 
    def __init__(self, r, rtypes, tickers, wgma_window_size, gma_window_size, agent_name, telegram_bot, chat_id, gamma): 
        ### some hyperparameters ### 
        self.r = r 
        self.rtypes = rtypes 
        self.tickers = tickers 
        self.assets = len(self.tickers) 
        self.iterations = 0 
        self.agent_name = agent_name 
        self.telegram_bot = telegram_bot 
        self.chat_id = chat_id 
        self.gamma = gamma 
        self.wgma_window_size = wgma_window_size
        self.gma_window_size = gma_window_size 
        
        ### get initial prices, volumes and dfs data ###
        text = "##### Very First Past Data Lookup For Model Initializations #####" 
        self.telegram_bot.sendMessage(chat_id=self.chat_id, text=text)  
        self.prices, self.dfs, self.returns = one_hour_price_lookup(self.tickers) 
        
        ### define base models ###
        self.pamr = PAMR(epsilon=0.5, C=500, variant=0, tickers=self.tickers, prices=self.prices)
        self.pmr = FBPROPHET_MR(epsilon=10, tickers=self.tickers, prices=self.prices, dfs=self.dfs, rtypes=self.rtypes)
        self.olmar5 = OLMAR(epsilon=10, window_size=5, tickers=self.tickers, prices=self.prices)
        self.olmar10 = OLMAR(epsilon=10, window_size=10, tickers=self.tickers, prices=self.prices)
        self.olmar15 = OLMAR(epsilon=10, window_size=15, tickers=self.tickers, prices=self.prices)
        self.olmar20 = OLMAR(epsilon=10, window_size=20, tickers=self.tickers, prices=self.prices)
        self.rmr5 = RMR(window_size=5, eps=10, tau=0.001, tickers=self.tickers, prices=self.prices)
        self.rmr10 = RMR(window_size=10, eps=10, tau=0.001, tickers=self.tickers, prices=self.prices) 
        self.rmr15 = RMR(window_size=15, eps=10, tau=0.001, tickers=self.tickers, prices=self.prices) 
        self.rmr20 = RMR(window_size=20, eps=10, tau=0.001, tickers=self.tickers, prices=self.prices) 
        self.wmamr5 = WMAMR(epsilon=0.5, C=500, variant=0, window_size=5, tickers=self.tickers, prices=self.prices)  
        self.wmamr10 = WMAMR(epsilon=0.5, C=500, variant=0, window_size=10, tickers=self.tickers, prices=self.prices)
        self.wmamr15 = WMAMR(epsilon=0.5, C=500, variant=0, window_size=15, tickers=self.tickers, prices=self.prices) 
        self.wmamr20 = WMAMR(epsilon=0.5, C=500, variant=0, window_size=20, tickers=self.tickers, prices=self.prices)
        self.rprt5 = RPRT(window_size=5, eps=50, theta=0.8, tickers=self.tickers, prices=self.prices) 
        self.rprt10 = RPRT(window_size=10, eps=50, theta=0.8, tickers=self.tickers, prices=self.prices)
        self.rprt15 = RPRT(window_size=15, eps=50, theta=0.8, tickers=self.tickers, prices=self.prices) 
        self.rprt20 = RPRT(window_size=20, eps=50, theta=0.8, tickers=self.tickers, prices=self.prices)
        self.cwmr = CWMR(eps=-0.5, confidence=0.95, tickers=self.tickers, prices=self.prices)
        self.tco_single = TCO_single(trx_fee_pct=self.gamma, eta=10, types="reverting", tickers=self.tickers, prices=self.prices) 
        self.tco_multi5 = TCO_multi(trx_fee_pct=self.gamma, eta=10, window_size=5, tickers=self.tickers, prices=self.prices)
        self.tco_multi10 = TCO_multi(trx_fee_pct=self.gamma, eta=10, window_size=10, tickers=self.tickers, prices=self.prices)
        self.tco_multi15 = TCO_multi(trx_fee_pct=self.gamma, eta=10, window_size=15, tickers=self.tickers, prices=self.prices)
        self.tco_multi20 = TCO_multi(trx_fee_pct=self.gamma, eta=10, window_size=20, tickers=self.tickers, prices=self.prices)
        self.ons = ONS(delta=0.125, beta=1.0, eta=0.0, assets=self.assets, tickers=self.tickers, prices=self.prices) 
        self.eg = EXPONENTIAL_GRADIENT(eta=0.05, tickers=self.tickers, prices=self.prices) 
        self.sr = SOFTMAX_REBALANCING(assets=self.assets, tickers=self.tickers, prices=self.prices) 
        self.uniform_crp = UNIFORM_CRP(assets=self.assets) 
        self.cash_agent = CASH_AGENT(assets=self.assets)
        self.fbp = FB_PROPHET(tickers=self.tickers, rtypes=self.rtypes, prices=self.prices, dfs=self.dfs) 
        
        ### define GMA model performance arrays ### 
        self.gma_pamr_history = []
        self.gma_pmr_history = [] 
        self.gma_olmar5_history = [] 
        self.gma_olmar10_history = [] 
        self.gma_olmar15_history = [] 
        self.gma_olmar20_history = [] 
        self.gma_rmr5_history = []
        self.gma_rmr10_history = [] 
        self.gma_rmr15_history = [] 
        self.gma_rmr20_history = [] 
        self.gma_wmamr5_history = [] 
        self.gma_wmamr10_history = [] 
        self.gma_wmamr15_history = [] 
        self.gma_wmamr20_history = [] 
        self.gma_rprt5_history = []
        self.gma_rprt10_history = [] 
        self.gma_rprt15_history = [] 
        self.gma_rprt20_history = [] 
        self.gma_cwmr_history = [] 
        self.gma_tco_single_history = [] 
        self.gma_tco_multi5_history = [] 
        self.gma_tco_multi10_history = [] 
        self.gma_tco_multi15_history = [] 
        self.gma_tco_multi20_history = [] 
        self.gma_ons_history = [] 
        self.gma_eg_history = [] 
        self.gma_sr_history = [] 
        self.gma_uniform_crp_history = []  
        self.gma_cash_agent_history = [] 
        self.gma_fbp_history = []
            
        ### define WGMA model performance arrays ###
        self.wgma_pamr_history = []
        self.wgma_pmr_history = [] 
        self.wgma_olmar5_history = [] 
        self.wgma_olmar10_history = [] 
        self.wgma_olmar15_history = [] 
        self.wgma_olmar20_history = [] 
        self.wgma_rmr5_history = []
        self.wgma_rmr10_history = [] 
        self.wgma_rmr15_history = [] 
        self.wgma_rmr20_history = [] 
        self.wgma_wmamr5_history = [] 
        self.wgma_wmamr10_history = [] 
        self.wgma_wmamr15_history = [] 
        self.wgma_wmamr20_history = [] 
        self.wgma_rprt5_history = []
        self.wgma_rprt10_history = [] 
        self.wgma_rprt15_history = [] 
        self.wgma_rprt20_history = [] 
        self.wgma_cwmr_history = [] 
        self.wgma_tco_single_history = [] 
        self.wgma_tco_multi5_history = [] 
        self.wgma_tco_multi10_history = [] 
        self.wgma_tco_multi15_history = [] 
        self.wgma_tco_multi20_history = [] 
        self.wgma_ons_history = [] 
        self.wgma_eg_history = [] 
        self.wgma_sr_history = [] 
        self.wgma_uniform_crp_history = [] 
        self.wgma_cash_agent_history = [] 
        self.wgma_fbp_history = [] 
            
        ### base model portfolios ###
        self.pamr_portfolio = [1/self.assets for i in range(self.assets)] 
        self.pmr_portfolio = [1/self.assets for i in range(self.assets)] 
        self.olmar5_portfolio = [1/self.assets for i in range(self.assets)]
        self.olmar10_portfolio = [1/self.assets for i in range(self.assets)]
        self.olmar15_portfolio = [1/self.assets for i in range(self.assets)]
        self.olmar20_portfolio = [1/self.assets for i in range(self.assets)]
        self.rmr5_portfolio = [1/self.assets for i in range(self.assets)]
        self.rmr10_portfolio = [1/self.assets for i in range(self.assets)] 
        self.rmr15_portfolio = [1/self.assets for i in range(self.assets)] 
        self.rmr20_portfolio = [1/self.assets for i in range(self.assets)] 
        self.wmamr5_portfolio = [1/self.assets for i in range(self.assets)]  
        self.wmamr10_portfolio = [1/self.assets for i in range(self.assets)] 
        self.wmamr15_portfolio = [1/self.assets for i in range(self.assets)] 
        self.wmamr20_portfolio = [1/self.assets for i in range(self.assets)] 
        self.rprt5_portfolio = [1/self.assets for i in range(self.assets)] 
        self.rprt10_portfolio = [1/self.assets for i in range(self.assets)]  
        self.rprt15_portfolio = [1/self.assets for i in range(self.assets)] 
        self.rprt20_portfolio = [1/self.assets for i in range(self.assets)] 
        self.cwmr_portfolio = [1/self.assets for i in range(self.assets)] 
        self.tco_single_portfolio = [1/self.assets for i in range(self.assets)] 
        self.tco_multi5_portfolio = [1/self.assets for i in range(self.assets)] 
        self.tco_multi10_portfolio = [1/self.assets for i in range(self.assets)] 
        self.tco_multi15_portfolio = [1/self.assets for i in range(self.assets)] 
        self.tco_multi20_portfolio = [1/self.assets for i in range(self.assets)] 
        self.ons_portfolio = [1/self.assets for i in range(self.assets)] 
        self.eg_portfolio = [1/self.assets for i in range(self.assets)] 
        self.sr_portfolio = [1/self.assets for i in range(self.assets)] 
        self.uniform_crp_portfolio = [1/self.assets for i in range(self.assets)] 
        self.fbp_portfolio = [1/self.assets for i in range(self.assets)]
        self.cash_agent_portfolio = [0 for i in range(self.assets)]
            
    def geometric_mean(self, x): 
        x = np.asarray(x) 
        return x.prod() ** (1/len(x)) 
    
    def weighted_geometric_mean(self, x, decay_factor=1.01): 
        x = np.asarray(x) 
        w = [1/np.power(decay_factor, i) for i in range(len(x))]
        prod = 1 
        for i in range(len(x)): 
            prod *= np.power(x[i], w[i]) 
        return prod ** (1/np.sum(w)) 
    
    def check_zero_portfolio(self, arr): 
        for i in range(len(arr)):
            if arr[i] != 0: 
                return False 
        return True 
    
    def predict(self): 
        if self.iterations > 0: 
            self.prices, self.dfs, self.returns = one_hour_price_lookup(self.tickers) 
            #################################################################
            ###                     Fill Up GMA ARR                       ###
            #################################################################
            self.gma_pamr_history.append(np.dot(self.pamr_portfolio, self.returns))
            self.gma_pmr_history.append(np.dot(self.pmr_portfolio, self.returns)) 
            self.gma_olmar5_history.append(np.dot(self.olmar5_portfolio, self.returns))
            self.gma_olmar10_history.append(np.dot(self.olmar10_portfolio, self.returns)) 
            self.gma_olmar15_history.append(np.dot(self.olmar15_portfolio, self.returns)) 
            self.gma_olmar20_history.append(np.dot(self.olmar20_portfolio, self.returns)) 
            self.gma_rmr5_history.append(np.dot(self.rmr5_portfolio, self.returns)) 
            self.gma_rmr10_history.append(np.dot(self.rmr10_portfolio, self.returns))
            self.gma_rmr15_history.append(np.dot(self.rmr15_portfolio, self.returns)) 
            self.gma_rmr20_history.append(np.dot(self.rmr20_portfolio, self.returns)) 
            self.gma_wmamr5_history.append(np.dot(self.wmamr5_portfolio, self.returns)) 
            self.gma_wmamr10_history.append(np.dot(self.wmamr10_portfolio, self.returns)) 
            self.gma_wmamr15_history.append(np.dot(self.wmamr15_portfolio, self.returns))
            self.gma_wmamr20_history.append(np.dot(self.wmamr20_portfolio, self.returns)) 
            self.gma_rprt5_history.append(np.dot(self.rprt5_portfolio, self.returns)) 
            self.gma_rprt10_history.append(np.dot(self.rprt10_portfolio, self.returns)) 
            self.gma_rprt15_history.append(np.dot(self.rprt15_portfolio, self.returns)) 
            self.gma_rprt20_history.append(np.dot(self.rprt20_portfolio, self.returns)) 
            self.gma_cwmr_history.append(np.dot(self.cwmr_portfolio, self.returns)) 
            self.gma_tco_single_history.append(np.dot(self.tco_single_portfolio, self.returns)) 
            self.gma_tco_multi5_history.append(np.dot(self.tco_multi5_portfolio, self.returns)) 
            self.gma_tco_multi10_history.append(np.dot(self.tco_multi10_portfolio, self.returns)) 
            self.gma_tco_multi15_history.append(np.dot(self.tco_multi15_portfolio, self.returns)) 
            self.gma_tco_multi20_history.append(np.dot(self.tco_multi20_portfolio, self.returns)) 
            self.gma_ons_history.append(np.dot(self.ons_portfolio, self.returns))
            self.gma_eg_history.append(np.dot(self.eg_portfolio, self.returns)) 
            self.gma_sr_history.append(np.dot(self.sr_portfolio, self.returns)) 
            self.gma_uniform_crp_history.append(np.dot(self.uniform_crp_portfolio, self.returns)) 
            self.gma_fbp_history.append(np.dot(self.fbp_portfolio, self.returns))
            self.gma_cash_agent_history.append(1.0) # cash agent gets 1 appended to its performance array all the time 
            
            if len(self.gma_pamr_history) > self.gma_window_size: 
                self.gma_pamr_history.pop(0) 
            if len(self.gma_pmr_history) > self.gma_window_size: 
                self.gma_pmr_history.pop(0) 
            if len(self.gma_olmar5_history) > self.gma_window_size: 
                self.gma_olmar5_history.pop(0) 
            if len(self.gma_olmar10_history) > self.gma_window_size: 
                self.gma_olmar10_history.pop(0)
            if len(self.gma_olmar15_history) > self.gma_window_size: 
                self.gma_olmar15_history.pop(0)
            if len(self.gma_olmar20_history) > self.gma_window_size: 
                self.gma_olmar20_history.pop(0)
            if len(self.gma_rmr5_history) > self.gma_window_size:  
                self.gma_rmr5_history.pop(0) 
            if len(self.gma_rmr10_history) > self.gma_window_size: 
                self.gma_rmr10_history.pop(0) 
            if len(self.gma_rmr15_history) > self.gma_window_size: 
                self.gma_rmr15_history.pop(0)
            if len(self.gma_rmr20_history) > self.gma_window_size: 
                self.gma_rmr20_history.pop(0) 
            if len(self.gma_wmamr5_history) > self.gma_window_size: 
                self.gma_wmamr5_history.pop(0)
            if len(self.gma_wmamr10_history) > self.gma_window_size: 
                self.gma_wmamr10_history.pop(0) 
            if len(self.gma_wmamr15_history) > self.gma_window_size: 
                self.gma_wmamr15_history.pop(0) 
            if len(self.gma_wmamr20_history) > self.gma_window_size: 
                self.gma_wmamr20_history.pop(0) 
            if len(self.gma_rprt5_history) > self.gma_window_size: 
                self.gma_rprt5_history.pop(0) 
            if len(self.gma_rprt10_history) > self.gma_window_size:  
                self.gma_rprt10_history.pop(0)
            if len(self.gma_rprt15_history) > self.gma_window_size:  
                self.gma_rprt15_history.pop(0) 
            if len(self.gma_rprt20_history) > self.gma_window_size:  
                self.gma_rprt20_history.pop(0) 
            if len(self.gma_cwmr_history) > self.gma_window_size: 
                self.gma_cwmr_history.pop(0) 
            if len(self.gma_tco_single_history) > self.gma_window_size: 
                self.gma_tco_single_history.pop(0) 
            if len(self.gma_tco_multi5_history) > self.gma_window_size:  
                self.gma_tco_multi5_history.pop(0) 
            if len(self.gma_tco_multi10_history) > self.gma_window_size:  
                self.gma_tco_multi10_history.pop(0) 
            if len(self.gma_tco_multi15_history) > self.gma_window_size: 
                self.gma_tco_multi15_history.pop(0) 
            if len(self.gma_tco_multi20_history) > self.gma_window_size:  
                self.gma_tco_multi20_history.pop(0)   
            if len(self.gma_ons_history) > self.gma_window_size: 
                self.gma_ons_history.pop(0)
            if len(self.gma_eg_history) > self.gma_window_size: 
                self.gma_eg_history.pop(0)
            if len(self.gma_sr_history) > self.gma_window_size:  
                self.gma_sr_history.pop(0) 
            if len(self.gma_uniform_crp_history) > self.gma_window_size: 
                self.gma_uniform_crp_history.pop(0)  
            if len(self.gma_fbp_history) > self.gma_window_size:  
                self.gma_fbp_history.pop(0) 
            if len(self.gma_cash_agent_history) > self.gma_window_size: 
                self.gma_cash_agent_history.pop(0) 
           
                
            #################################################################
            ###                    Fill Up WGMA ARR                       ###
            #################################################################
            self.wgma_pamr_history.append(np.dot(self.pamr_portfolio, self.returns))
            self.wgma_pmr_history.append(np.dot(self.pmr_portfolio, self.returns)) 
            self.wgma_olmar5_history.append(np.dot(self.olmar5_portfolio, self.returns))
            self.wgma_olmar10_history.append(np.dot(self.olmar10_portfolio, self.returns)) 
            self.wgma_olmar15_history.append(np.dot(self.olmar15_portfolio, self.returns)) 
            self.wgma_olmar20_history.append(np.dot(self.olmar20_portfolio, self.returns)) 
            self.wgma_rmr5_history.append(np.dot(self.rmr5_portfolio, self.returns)) 
            self.wgma_rmr10_history.append(np.dot(self.rmr10_portfolio, self.returns))
            self.wgma_rmr15_history.append(np.dot(self.rmr15_portfolio, self.returns)) 
            self.wgma_rmr20_history.append(np.dot(self.rmr20_portfolio, self.returns)) 
            self.wgma_wmamr5_history.append(np.dot(self.wmamr5_portfolio, self.returns)) 
            self.wgma_wmamr10_history.append(np.dot(self.wmamr10_portfolio, self.returns)) 
            self.wgma_wmamr15_history.append(np.dot(self.wmamr15_portfolio, self.returns))
            self.wgma_wmamr20_history.append(np.dot(self.wmamr20_portfolio, self.returns)) 
            self.wgma_rprt5_history.append(np.dot(self.rprt5_portfolio, self.returns)) 
            self.wgma_rprt10_history.append(np.dot(self.rprt10_portfolio, self.returns)) 
            self.wgma_rprt15_history.append(np.dot(self.rprt15_portfolio, self.returns)) 
            self.wgma_rprt20_history.append(np.dot(self.rprt20_portfolio, self.returns)) 
            self.wgma_cwmr_history.append(np.dot(self.cwmr_portfolio, self.returns)) 
            self.wgma_tco_single_history.append(np.dot(self.tco_single_portfolio, self.returns)) 
            self.wgma_tco_multi5_history.append(np.dot(self.tco_multi5_portfolio, self.returns)) 
            self.wgma_tco_multi10_history.append(np.dot(self.tco_multi10_portfolio, self.returns)) 
            self.wgma_tco_multi15_history.append(np.dot(self.tco_multi15_portfolio, self.returns)) 
            self.wgma_tco_multi20_history.append(np.dot(self.tco_multi20_portfolio, self.returns)) 
            self.wgma_ons_history.append(np.dot(self.ons_portfolio, self.returns))
            self.wgma_eg_history.append(np.dot(self.eg_portfolio, self.returns)) 
            self.wgma_sr_history.append(np.dot(self.sr_portfolio, self.returns)) 
            self.wgma_uniform_crp_history.append(np.dot(self.uniform_crp_portfolio, self.returns)) 
            self.wgma_fbp_history.append(np.dot(self.fbp_portfolio, self.returns))
            self.wgma_cash_agent_history.append(1.0) # cash agent gets 1 appended to its performance array all the time 
            
            if len(self.wgma_pamr_history) > self.wgma_window_size: 
                self.wgma_pamr_history.pop(0) 
            if len(self.wgma_pmr_history) > self.wgma_window_size: 
                self.wgma_pmr_history.pop(0) 
            if len(self.wgma_olmar5_history) > self.wgma_window_size: 
                self.wgma_olmar5_history.pop(0) 
            if len(self.wgma_olmar10_history) > self.wgma_window_size: 
                self.wgma_olmar10_history.pop(0)
            if len(self.wgma_olmar15_history) > self.wgma_window_size: 
                self.wgma_olmar15_history.pop(0)
            if len(self.wgma_olmar20_history) > self.wgma_window_size: 
                self.wgma_olmar20_history.pop(0)
            if len(self.wgma_rmr5_history) > self.wgma_window_size:  
                self.wgma_rmr5_history.pop(0) 
            if len(self.wgma_rmr10_history) > self.wgma_window_size: 
                self.wgma_rmr10_history.pop(0) 
            if len(self.wgma_rmr15_history) > self.wgma_window_size: 
                self.wgma_rmr15_history.pop(0)
            if len(self.wgma_rmr20_history) > self.wgma_window_size: 
                self.wgma_rmr20_history.pop(0) 
            if len(self.wgma_wmamr5_history) > self.wgma_window_size: 
                self.wgma_wmamr5_history.pop(0)
            if len(self.wgma_wmamr10_history) > self.wgma_window_size: 
                self.wgma_wmamr10_history.pop(0) 
            if len(self.wgma_wmamr15_history) > self.wgma_window_size: 
                self.wgma_wmamr15_history.pop(0) 
            if len(self.wgma_wmamr20_history) > self.wgma_window_size: 
                self.wgma_wmamr20_history.pop(0) 
            if len(self.wgma_rprt5_history) > self.wgma_window_size: 
                self.wgma_rprt5_history.pop(0) 
            if len(self.wgma_rprt10_history) > self.wgma_window_size:  
                self.wgma_rprt10_history.pop(0)
            if len(self.wgma_rprt15_history) > self.wgma_window_size:  
                self.wgma_rprt15_history.pop(0) 
            if len(self.wgma_rprt20_history) > self.wgma_window_size:  
                self.wgma_rprt20_history.pop(0) 
            if len(self.wgma_cwmr_history) > self.wgma_window_size: 
                self.wgma_cwmr_history.pop(0) 
            if len(self.wgma_tco_single_history) > self.wgma_window_size: 
                self.wgma_tco_single_history.pop(0) 
            if len(self.wgma_tco_multi5_history) > self.wgma_window_size:  
                self.wgma_tco_multi5_history.pop(0) 
            if len(self.wgma_tco_multi10_history) > self.wgma_window_size:  
                self.wgma_tco_multi10_history.pop(0) 
            if len(self.wgma_tco_multi15_history) > self.wgma_window_size: 
                self.wgma_tco_multi15_history.pop(0) 
            if len(self.wgma_tco_multi20_history) > self.wgma_window_size:  
                self.wgma_tco_multi20_history.pop(0)   
            if len(self.wgma_ons_history) > self.wgma_window_size: 
                self.wgma_ons_history.pop(0)
            if len(self.wgma_eg_history) > self.wgma_window_size: 
                self.wgma_eg_history.pop(0)
            if len(self.wgma_sr_history) > self.wgma_window_size:  
                self.wgma_sr_history.pop(0) 
            if len(self.wgma_uniform_crp_history) > self.wgma_window_size: 
                self.wgma_uniform_crp_history.pop(0)  
            if len(self.wgma_fbp_history) > self.wgma_window_size:  
                self.wgma_fbp_history.pop(0) 
            if len(self.wgma_cash_agent_history) > self.wgma_window_size: 
                self.wgma_cash_agent_history.pop(0) 
        
        #################################################################
        ###                    Make predictions                       ###
        #################################################################
        self.pamr_portfolio = self.pamr.predict(self.pamr_portfolio)
        self.pmr_portfolio = self.pmr.predict(self.pmr_portfolio) 
        self.olmar5_portfolio = self.olmar5.predict(self.olmar5_portfolio) 
        self.olmar10_portfolio = self.olmar10.predict(self.olmar10_portfolio) 
        self.olmar15_portfolio = self.olmar15.predict(self.olmar15_portfolio) 
        self.olmar20_portfolio = self.olmar20.predict(self.olmar20_portfolio) 
        self.rmr5_portfolio = self.rmr5.predict(self.rmr5_portfolio)
        self.rmr10_portfolio = self.rmr10.predict(self.rmr10_portfolio) 
        self.rmr15_portfolio = self.rmr15.predict(self.rmr15_portfolio) 
        self.rmr20_portfolio = self.rmr20.predict(self.rmr20_portfolio) 
        self.wmamr5_portfolio = self.wmamr5.predict(self.wmamr5_portfolio) 
        self.wmamr10_portfolio = self.wmamr10.predict(self.wmamr10_portfolio) 
        self.wmamr15_portfolio = self.wmamr15.predict(self.wmamr15_portfolio) 
        self.wmamr20_portfolio = self.wmamr20.predict(self.wmamr20_portfolio) 
        self.rprt5_portfolio = self.rprt5.predict(self.rprt5_portfolio) 
        self.rprt10_portfolio = self.rprt10.predict(self.rprt10_portfolio) 
        self.rprt15_portfolio = self.rprt15.predict(self.rprt15_portfolio) 
        self.rprt20_portfolio = self.rprt20.predict(self.rprt20_portfolio) 
        self.cwmr_portfolio = self.cwmr.predict(self.cwmr_portfolio)
        self.tco_single_portfolio = self.tco_single.predict(self.tco_single_portfolio) 
        self.tco_multi5_portfolio = self.tco_multi5.predict(self.tco_multi5_portfolio) 
        self.tco_multi10_portfolio = self.tco_multi10.predict(self.tco_multi10_portfolio)
        self.tco_multi15_portfolio = self.tco_multi15.predict(self.tco_multi15_portfolio) 
        self.tco_multi20_portfolio = self.tco_multi20.predict(self.tco_multi20_portfolio) 
        self.ons_portfolio = self.ons.predict(self.ons_portfolio) 
        self.eg_portfolio = self.eg.predict(self.eg_portfolio) 
        self.sr_portfolio = self.sr.predict(self.sr_portfolio) 
        self.uniform_crp_portfolio = self.uniform_crp.predict(self.uniform_crp_portfolio)
        self.fbp_portfolio = self.fbp.predict(self.fbp_portfolio) 
        self.cash_agent_portfolio = self.cash_agent.predict(self.cash_agent_portfolio) 
      
            
        
      
        model_portfolios = [self.pamr_portfolio, 
                            self.pmr_portfolio, 
                            self.olmar5_portfolio, 
                            self.olmar10_portfolio, 
                            self.olmar15_portfolio, 
                            self.olmar20_portfolio, 
                            self.rmr5_portfolio,
                            self.rmr10_portfolio,
                            self.rmr15_portfolio, 
                            self.rmr20_portfolio, 
                            self.wmamr5_portfolio, 
                            self.wmamr10_portfolio, 
                            self.wmamr15_portfolio, 
                            self.wmamr20_portfolio, 
                            self.rprt5_portfolio, 
                            self.rprt10_portfolio,  
                            self.rprt15_portfolio, 
                            self.rprt20_portfolio,  
                            self.cwmr_portfolio,  
                            self.tco_single_portfolio, 
                            self.tco_multi5_portfolio, 
                            self.tco_multi10_portfolio,  
                            self.tco_multi15_portfolio,  
                            self.tco_multi20_portfolio, 
                            self.ons_portfolio, 
                            self.eg_portfolio, 
                            self.sr_portfolio, 
                            self.uniform_crp_portfolio, 
                            self.fbp_portfolio, 
                            self.cash_agent_portfolio] 
                
                
        model_names = ["PAMR",
                        "PMR", 
                        "OLMAR5", 
                        "OLMAR10", 
                        "OLMAR15", 
                        "OLMAR20", 
                        "RMR5", 
                        "RMR10", 
                        "RMR15", 
                        "RMR20", 
                        "WMAMR5", 
                        "WMAMR10", 
                        "WMAMR15", 
                        "WMAMR20", 
                        "RPRT5", 
                        "RPRT10", 
                        "RPRT15", 
                        "RPRT20", 
                        "CWMR", 
                        "TCO_SINGLE", 
                        "TCO_MULTI5", 
                        "TCO_MULTI10", 
                        "TCO_MULTI15", 
                        "TCO_MULTI20", 
                        "ONS", 
                        "EXPONENTIAL GRADIENT", 
                        "SOFTMAX REBALANCING", 
                        "UNIFORM CRP", 
                        "FBP", 
                        "CASH_AGENT"]
            
        if self.iterations == 0: 
            # choose ONS for first trade  
            text = "First trade for " + self.agent_name + ". Choosing ONS" 
            self.telegram_bot.sendMessage(chat_id=self.chat_id, text=text)
            self.iterations += 1 
            return self.ons_portfolio 
        else: 
            self.iterations += 1 
            #################################################################
            ###      decide cash agent based on wgma model selector       ###
            #################################################################
            wgma_pamr = self.weighted_geometric_mean(self.wgma_pamr_history) 
            wgma_pmr = self.weighted_geometric_mean(self.wgma_pmr_history) 
            wgma_olmar5 = self.weighted_geometric_mean(self.wgma_olmar5_history) 
            wgma_olmar10 = self.weighted_geometric_mean(self.wgma_olmar10_history) 
            wgma_olmar15 = self.weighted_geometric_mean(self.wgma_olmar15_history) 
            wgma_olmar20 = self.weighted_geometric_mean(self.wgma_olmar20_history) 
            wgma_rmr5 = self.weighted_geometric_mean(self.wgma_rmr5_history) 
            wgma_rmr10 = self.weighted_geometric_mean(self.wgma_rmr10_history) 
            wgma_rmr15 = self.weighted_geometric_mean(self.wgma_rmr15_history) 
            wgma_rmr20 = self.weighted_geometric_mean(self.wgma_rmr20_history) 
            wgma_wmamr5 = self.weighted_geometric_mean(self.wgma_wmamr5_history) 
            wgma_wmamr10 = self.weighted_geometric_mean(self.wgma_wmamr10_history) 
            wgma_wmamr15 = self.weighted_geometric_mean(self.wgma_wmamr15_history) 
            wgma_wmamr20 = self.weighted_geometric_mean(self.wgma_wmamr20_history) 
            wgma_rprt5 = self.weighted_geometric_mean(self.wgma_rprt5_history) 
            wgma_rprt10 = self.weighted_geometric_mean(self.wgma_rprt10_history)  
            wgma_rprt15 = self.weighted_geometric_mean(self.wgma_rprt15_history) 
            wgma_rprt20 = self.weighted_geometric_mean(self.wgma_rprt20_history) 
            wgma_cwmr = self.weighted_geometric_mean(self.wgma_cwmr_history)  
            wgma_tco_single = self.weighted_geometric_mean(self.wgma_tco_single_history) 
            wgma_tco_multi5 = self.weighted_geometric_mean(self.wgma_tco_multi5_history) 
            wgma_tco_multi10 = self.weighted_geometric_mean(self.wgma_tco_multi10_history) 
            wgma_tco_multi15 = self.weighted_geometric_mean(self.wgma_tco_multi15_history) 
            wgma_tco_multi20 = self.weighted_geometric_mean(self.wgma_tco_multi20_history) 
            wgma_ons = self.weighted_geometric_mean(self.wgma_ons_history) 
            wgma_eg = self.weighted_geometric_mean(self.wgma_eg_history) 
            wgma_sr = self.weighted_geometric_mean(self.wgma_sr_history) 
            wgma_uniform_crp = self.weighted_geometric_mean(self.wgma_uniform_crp_history)
            wgma_fbp = self.weighted_geometric_mean(self.wgma_fbp_history) 
            wgma_cash_agent = self.weighted_geometric_mean(self.wgma_cash_agent_history, decay_factor=1.0)  

            wgma_profits = [wgma_pamr,
                            wgma_pmr, 
                            wgma_olmar5,
                            wgma_olmar10, 
                            wgma_olmar15, 
                            wgma_olmar20, 
                            wgma_rmr5,
                            wgma_rmr10, 
                            wgma_rmr15, 
                            wgma_rmr20, 
                            wgma_wmamr5,
                            wgma_wmamr10, 
                            wgma_wmamr15, 
                            wgma_wmamr20, 
                            wgma_rprt5, 
                            wgma_rprt10, 
                            wgma_rprt15, 
                            wgma_rprt20, 
                            wgma_cwmr, 
                            wgma_tco_single,
                            wgma_tco_multi5, 
                            wgma_tco_multi10, 
                            wgma_tco_multi15, 
                            wgma_tco_multi20, 
                            wgma_ons, 
                            wgma_eg, 
                            wgma_sr, 
                            wgma_uniform_crp, 
                            wgma_fbp, 
                            wgma_cash_agent] 
            
            best_wgma_profits, best_wgma_idx = np.max(wgma_profits), np.argmax(wgma_profits) 
            if model_names[best_wgma_idx] == "CASH_AGENT": 
                text = self.agent_name + " choosing cash agent." 
                self.telegram_bot.sendMessage(chat_id=self.chat_id, text=text)
                return model_portfolios[best_wgma_idx] 
            else: 
                #################################################################
                ###      decide base mode based on gma model selector         ###
                #################################################################
                gma_pamr = self.geometric_mean(self.gma_pamr_history) 
                gma_pmr = self.geometric_mean(self.gma_pmr_history) 
                gma_olmar5 = self.geometric_mean(self.gma_olmar5_history) 
                gma_olmar10 = self.geometric_mean(self.gma_olmar10_history) 
                gma_olmar15 = self.geometric_mean(self.gma_olmar15_history) 
                gma_olmar20 = self.geometric_mean(self.gma_olmar20_history) 
                gma_rmr5 = self.geometric_mean(self.gma_rmr5_history) 
                gma_rmr10 = self.geometric_mean(self.gma_rmr10_history) 
                gma_rmr15 = self.geometric_mean(self.gma_rmr15_history) 
                gma_rmr20 = self.geometric_mean(self.gma_rmr20_history) 
                gma_wmamr5 = self.geometric_mean(self.gma_wmamr5_history) 
                gma_wmamr10 = self.geometric_mean(self.gma_wmamr10_history) 
                gma_wmamr15 = self.geometric_mean(self.gma_wmamr15_history) 
                gma_wmamr20 = self.geometric_mean(self.gma_wmamr20_history) 
                gma_rprt5 = self.geometric_mean(self.gma_rprt5_history) 
                gma_rprt10 = self.geometric_mean(self.gma_rprt10_history)  
                gma_rprt15 = self.geometric_mean(self.gma_rprt15_history) 
                gma_rprt20 = self.geometric_mean(self.gma_rprt20_history) 
                gma_cwmr = self.geometric_mean(self.gma_cwmr_history)  
                gma_tco_single = self.geometric_mean(self.gma_tco_single_history) 
                gma_tco_multi5 = self.geometric_mean(self.gma_tco_multi5_history) 
                gma_tco_multi10 = self.geometric_mean(self.gma_tco_multi10_history) 
                gma_tco_multi15 = self.geometric_mean(self.gma_tco_multi15_history) 
                gma_tco_multi20 = self.geometric_mean(self.gma_tco_multi20_history) 
                gma_ons = self.geometric_mean(self.gma_ons_history) 
                gma_eg = self.geometric_mean(self.gma_eg_history) 
                gma_sr = self.geometric_mean(self.gma_sr_history) 
                gma_uniform_crp = self.geometric_mean(self.gma_uniform_crp_history)
                gma_fbp = self.geometric_mean(self.gma_fbp_history) 
                gma_cash_agent = self.geometric_mean(self.gma_cash_agent_history) 
                    
                
                gma_profits = [gma_pamr,
                               gma_pmr, 
                               gma_olmar5,
                               gma_olmar10, 
                               gma_olmar15, 
                               gma_olmar20, 
                               gma_rmr5,
                               gma_rmr10, 
                               gma_rmr15, 
                               gma_rmr20, 
                               gma_wmamr5,
                               gma_wmamr10, 
                               gma_wmamr15, 
                               gma_wmamr20, 
                               gma_rprt5, 
                               gma_rprt10, 
                               gma_rprt15, 
                               gma_rprt20, 
                               gma_cwmr, 
                               gma_tco_single,
                               gma_tco_multi5, 
                               gma_tco_multi10, 
                               gma_tco_multi15, 
                               gma_tco_multi20, 
                               gma_ons, 
                               gma_eg, 
                               gma_sr, 
                               gma_uniform_crp, 
                               gma_fbp, 
                               gma_cash_agent]
                
                best_gma_profits, best_gma_idx = np.max(gma_profits), np.argmax(gma_profits) 
                text = self.agent_name + " choosing " + model_names[best_gma_idx]   
                self.telegram_bot.sendMessage(chat_id=self.chat_id, text=text)
                # print("{} choosing {}".format(self.agent_name, model_name[best_gma_idx])) 
                return model_portfolios[best_gma_idx]
    
