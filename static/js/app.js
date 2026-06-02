// ===== APP ROUTER & STATE =====
let appData = null;
let currentCurrency = 'INR';
const exchangeRates = { INR: 1, USD: 0.011, EUR: 0.0091 };
const currencySymbols = { INR: '₹', USD: '$', EUR: '€' };
const currencyFlags = { INR: '🇮🇳', USD: '🇺🇸', EUR: '🇪🇺' };

let userCards = JSON.parse(localStorage.getItem('finAI_cards')) || [];
let activeCardIndex = 0;
let editingCardIndex = -1; // -1 means adding new, otherwise editing index
let userActivities = JSON.parse(localStorage.getItem('finAI_activities')) || [];

const App = {
  init() {
    // Load profile if exists
    const profile = JSON.parse(localStorage.getItem('finAI_profile'));
    if (profile) {
      setTimeout(() => {
        const iInc = document.getElementById('inp-income');
        if (iInc) {
          iInc.value = profile.inc || '';
          document.getElementById('inp-expenses').value = profile.exp || '';
          document.getElementById('inp-savings').value = profile.sav || '';
          document.getElementById('inp-debt').value = profile.debt || '';
        }
        
        // Auto-load if we have profile data
        if (window.location.hash !== '#login' && window.location.hash !== '') {
            App.triggerUpdate(false, null);
        }
      }, 100);
    }

    window.addEventListener('hashchange', App.route);
    if (!window.location.hash) window.location.hash = '#login';
    App.route();

    // Global Listeners
    document.getElementById('login-form')?.addEventListener('submit', async e => {
      e.preventDefault();
      const email = document.getElementById('login-email').value;
      const password = document.getElementById('login-password').value;
      const btn = document.getElementById('login-btn');
      const originalText = btn.textContent;
      
      btn.innerHTML = '<span class="spinner"></span> Logging in...';
      btn.disabled = true;

      try {
        const response = await fetch('/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email, password })
        });
        
        const result = await response.json();
        if (response.ok && result.success) {
          // Store user
          localStorage.setItem('finAI_user', JSON.stringify(result.user));
          
          // Restore state
          const state = result.state;
          if (state) {
            if (state.profile) {
              localStorage.setItem('finAI_profile', JSON.stringify(state.profile));
            } else {
              localStorage.removeItem('finAI_profile');
            }
            userCards = state.cards || [];
            localStorage.setItem('finAI_cards', JSON.stringify(userCards));
            
            userActivities = state.activities || [];
            localStorage.setItem('finAI_activities', JSON.stringify(userActivities));
            
            localStorage.setItem('finAI_goals', JSON.stringify(state.goals || []));
          } else {
            localStorage.removeItem('finAI_profile');
            localStorage.removeItem('finAI_cards');
            localStorage.removeItem('finAI_activities');
            localStorage.removeItem('finAI_goals');
            userCards = [];
            userActivities = [];
          }
          
          // Load input forms
          const profile = JSON.parse(localStorage.getItem('finAI_profile'));
          if (profile) {
            const iInc = document.getElementById('inp-income');
            if (iInc) {
              iInc.value = profile.inc || '';
              document.getElementById('inp-expenses').value = profile.exp || '';
              document.getElementById('inp-savings').value = profile.sav || '';
              document.getElementById('inp-debt').value = profile.debt || '';
            }
            const dInc = document.getElementById('dash-inp-income');
            if (dInc) {
              dInc.value = profile.inc || '';
              document.getElementById('dash-inp-expenses').value = profile.exp || '';
              document.getElementById('dash-inp-savings').value = profile.sav || '';
              document.getElementById('dash-inp-debt').value = profile.debt || '';
            }
            
            // Auto update dashboard metrics
            await App.triggerUpdate(false, null);
            window.location.hash = '#dashboard';
          } else {
            // Reset input values
            const iInc = document.getElementById('inp-income');
            if (iInc) {
              iInc.value = '';
              document.getElementById('inp-expenses').value = '';
              document.getElementById('inp-savings').value = '';
              document.getElementById('inp-debt').value = '';
            }
            window.location.hash = '#input';
          }
        } else {
          alert(result.error || 'Invalid login details.');
        }
      } catch (err) {
        console.error(err);
        alert('Server error, please try again.');
      } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
      }
    });

    document.getElementById('signup-form')?.addEventListener('submit', async e => {
      e.preventDefault();
      const email = document.getElementById('signup-email').value;
      const password = document.getElementById('signup-password').value;
      const confirmPassword = document.getElementById('signup-confirm-password').value;
      const btn = document.getElementById('signup-btn');
      
      if (password !== confirmPassword) {
        alert('Passwords do not match.');
        return;
      }
      
      const originalText = btn.textContent;
      btn.innerHTML = '<span class="spinner"></span> Creating...';
      btn.disabled = true;

      try {
        const response = await fetch('/signup', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email, password })
        });
        
        const result = await response.json();
        if (response.ok && result.success) {
          alert('Account created successfully! Please login.');
          window.location.hash = '#login';
          document.getElementById('signup-form').reset();
        } else {
          alert(result.error || 'Failed to create account.');
        }
      } catch (err) {
        console.error(err);
        alert('Server error, please try again.');
      } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
      }
    });

    document.getElementById('toggle-signup-password')?.addEventListener('click', function () {
      const pwd = document.getElementById('signup-password');
      if (pwd.type === 'password') {
        pwd.type = 'text';
        this.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path><line x1="1" y1="1" x2="23" y2="23"></line></svg>';
      } else {
        pwd.type = 'password';
        this.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>';
      }
    });

    // Password visibility toggle
    document.getElementById('toggle-password')?.addEventListener('click', function () {
      const pwd = document.getElementById('login-password');
      if (pwd.type === 'password') {
        pwd.type = 'text';
        this.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path><line x1="1" y1="1" x2="23" y2="23"></line></svg>';
      } else {
        pwd.type = 'password';
        this.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>';
      }
    });

    document.getElementById('setup-form')?.addEventListener('submit', e => {
      e.preventDefault();
      App.triggerUpdate(false, document.getElementById('analyze-btn'));
    });
    document.getElementById('dash-analyze-btn')?.addEventListener('click', e => { e.preventDefault(); App.triggerUpdate(true, e.target); });

    // Projections Toggle
    document.getElementById('proj-btn-6')?.addEventListener('click', () => {
      document.getElementById('proj-btn-12').classList.remove('active');
      document.getElementById('proj-btn-6').classList.add('active');
      Charts.lineChart('proj-chart', appData.forecast6, { lines: [{ key: 'savings', color: '#10B981' }, { key: 'income', color: '#3B82F6' }] });
    });
    document.getElementById('proj-btn-12')?.addEventListener('click', () => {
      document.getElementById('proj-btn-6').classList.remove('active');
      document.getElementById('proj-btn-12').classList.add('active');
      Charts.lineChart('proj-chart', appData.forecast12, { lines: [{ key: 'savings', color: '#10B981' }, { key: 'income', color: '#3B82F6' }] });
    });

    // Goals feature
    document.getElementById('toggle-goal-btn')?.addEventListener('click', () => {
      const form = document.getElementById('add-goal-form');
      form.style.display = form.style.display === 'none' ? 'block' : 'none';
    });

    document.getElementById('save-goal-btn')?.addEventListener('click', async () => {
      const name = document.getElementById('goal-name').value;
      const target = Number(document.getElementById('goal-target').value);
      const current = Number(document.getElementById('goal-current').value);
      if (name && target) {
        appData.goals.push({
          id: Date.now(),
          name: name,
          target: target,
          current: current,
          months: Math.ceil((target - current) / Math.max(appData.surplus, 1)),
          status: current >= target ? 'Completed' : 'On Track'
        });
        localStorage.setItem('finAI_goals', JSON.stringify(appData.goals));
        App.logActivity('New Goal Created', '🎯', target, 'Completed', name, false);
        Render.activitiesList('dash-activities', 5);
        document.getElementById('add-goal-form').style.display = 'none';

        // Reset fields
        document.getElementById('goal-name').value = '';
        document.getElementById('goal-target').value = '';
        document.getElementById('goal-current').value = '';

        App.renderPage('#goals');
        await App.syncState();
      }
    });

    // Wallets & Currency Switcher
    document.querySelectorAll('.currency-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const selected = e.currentTarget.getAttribute('data-currency');
        if (selected === currentCurrency) return;
        currentCurrency = selected;

        // Reset styles for all
        document.querySelectorAll('.currency-btn').forEach(b => {
          b.style.background = ''; b.style.color = ''; b.style.border = '';
        });

        // Set active style
        e.currentTarget.style.background = 'linear-gradient(135deg,rgba(59,130,246,0.1),rgba(59,130,246,0.05))';
        e.currentTarget.style.color = 'var(--secondary)';
        e.currentTarget.style.border = '1px solid rgba(59,130,246,0.2)';

        // Update pill
        document.getElementById('active-currency-pill').innerHTML = `${currentCurrency} ${currencyFlags[currentCurrency]}`;

        App.renderPage('#dashboard');
      });
    });

    // (Old Transfer/Request listeners removed to prevent duplicate popups)

    // Card Modal Listeners
    const cardModal = document.getElementById('card-modal');
    const transactionModal = document.getElementById('transaction-modal');
    const transactionForm = document.getElementById('transaction-form');

    // Transaction Modal Logic
    document.getElementById('open-transaction-modal-btn')?.addEventListener('click', () => {
      if (transactionForm) transactionForm.reset();
      if (transactionModal) transactionModal.style.display = 'flex';
    });

    document.getElementById('close-transaction-modal')?.addEventListener('click', () => {
      if (transactionModal) transactionModal.style.display = 'none';
    });

    transactionForm?.addEventListener('submit', async (e) => {
      e.preventDefault();
      const typeRadio = document.querySelector('input[name="trans-type"]:checked');
      if (!typeRadio) return;
      const type = typeRadio.value;
      const amount = Number(document.getElementById('trans-amount').value);
      const catVal = document.getElementById('trans-category').value;
      const [category, icon] = catVal.split('|');
      const note = document.getElementById('trans-note').value;
      
      App.logActivity(category, icon, amount, 'Completed', note, type === 'expense');
      
      // Update inputs on dashboard
      const expInput = document.getElementById('dash-inp-expenses');
      const incInput = document.getElementById('dash-inp-income');
      const savInput = document.getElementById('dash-inp-savings');
      
      if (type === 'expense') {
        if (expInput) expInput.value = Number(expInput.value) + amount;
        if (savInput) savInput.value = Math.max(0, Number(savInput.value) - amount);
      } else {
        if (incInput) incInput.value = Number(incInput.value) + amount;
        if (savInput) savInput.value = Number(savInput.value) + amount;
      }
      
      if (transactionModal) transactionModal.style.display = 'none';
      
      // Re-trigger analysis dynamically
      await App.triggerUpdate(true, null);
    });
    document.getElementById('add-card-btn')?.addEventListener('click', () => {
      editingCardIndex = -1;
      document.getElementById('card-modal-title').textContent = 'Add New Card';
      document.getElementById('card-form').reset();
      const typeContainer = document.getElementById('card-type')?.parentElement;
      if (typeContainer) typeContainer.style.display = 'block';
      cardModal.style.display = 'flex';
    });

    document.getElementById('close-card-modal')?.addEventListener('click', () => {
      cardModal.style.display = 'none';
    });

    document.getElementById('card-form')?.addEventListener('submit', async (e) => {
      e.preventDefault();
      const name = document.getElementById('card-name').value;
      const number = document.getElementById('card-number').value;
      const expiry = document.getElementById('card-expiry').value;

      if (editingCardIndex === -1) {
        const typeSelect = document.getElementById('card-type');
        const type = typeSelect.value;
        const bank = typeSelect.options[typeSelect.selectedIndex].text;
        userCards.push({ name, number, expiry, type, bank });
        activeCardIndex = userCards.length - 1;
      } else {
        userCards[editingCardIndex] = { ...userCards[editingCardIndex], name, number, expiry };
      }
      
      const typeSelect = document.getElementById('card-type');
      const bank = typeSelect.options[typeSelect.selectedIndex].text;
      if (editingCardIndex === -1) {
        App.logActivity('Card Added', '💳', 0, 'Completed', `${bank} ending in ${number.slice(-4)}`, false);
      }
      
      localStorage.setItem('finAI_cards', JSON.stringify(userCards));
      cardModal.style.display = 'none';
      App.renderPage('#dashboard');
      await App.syncState();
    });

    // Card Number Formatting
    document.getElementById('card-number')?.addEventListener('input', (e) => {
      let val = e.target.value.replace(/\s+/g, '').replace(/[^0-9]/gi, '');
      let matches = val.match(/\d{4,16}/g);
      let match = matches && matches[0] || '';
      let parts = [];
      for (let i = 0, len = match.length; i < len; i += 4) {
        parts.push(match.slice(i, i + 4));
      }
      if (parts.length) {
        e.target.value = parts.join(' ');
      }
    });

    // Card Expiry Formatting
    document.getElementById('card-expiry')?.addEventListener('input', (e) => {
      let val = e.target.value.replace(/\s+/g, '').replace(/[^0-9]/gi, '');
      if (val.length >= 2) {
        e.target.value = val.slice(0, 2) + '/' + val.slice(2, 4);
      }
    });

    // Transfer / Request Logic
    document.getElementById('btn-transfer')?.addEventListener('click', async () => {
      const amt = prompt('Enter amount to transfer (₹):');
      if (amt && !isNaN(amt) && Number(amt) > 0) {
        const val = Number(amt);
        const savInput = document.getElementById('dash-inp-savings');
        if (savInput) savInput.value = Math.max(0, Number(savInput.value) - val);
        App.logActivity('Transfer', '💸', val, 'Completed', 'Outgoing transfer', true);
        await App.triggerUpdate(true, null);
      }
    });

    document.getElementById('btn-request')?.addEventListener('click', async () => {
      const amt = prompt('Enter amount to request (₹):');
      if (amt && !isNaN(amt) && Number(amt) > 0) {
        const val = Number(amt);
        const savInput = document.getElementById('dash-inp-savings');
        if (savInput) savInput.value = Number(savInput.value) + val;
        App.logActivity('Request', '📥', val, 'Pending', 'Incoming request', false);
        await App.triggerUpdate(true, null);
      }
    });

    // View All Modal
    document.getElementById('view-all-activities-btn')?.addEventListener('click', () => {
      document.getElementById('all-activities-modal').style.display = 'flex';
      Render.activitiesList('all-activities-list', 100);
    });
    
    document.getElementById('close-all-activities-modal')?.addEventListener('click', () => {
      document.getElementById('all-activities-modal').style.display = 'none';
    });
  },

  logActivity(type, icon, amount, status, note, isExpense) {
    if (userActivities === null && appData) userActivities = appData.activities;
    if (!userActivities) userActivities = [];
    userActivities.unshift({
        type, icon, amount, status, note,
        date: new Date().toLocaleString('en-GB', { day: 'numeric', month: 'short', hour: 'numeric', minute: '2-digit', hour12: true }),
        isExpense
    });
    if (appData) appData.activities = userActivities;
    localStorage.setItem('finAI_activities', JSON.stringify(userActivities));
    // Immediately persist activity to DB (belt-and-suspenders safety)
    App.syncState();
  },

  route() {
    const hash = window.location.hash || '#login';

    // Hide all pages
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));

    // Shell visibility
    const shell = document.getElementById('app-shell');
    if (hash === '#login' || hash === '#signup' || hash === '#input') {
      if (hash === '#login') {
        // Final sync before clearing – fire the request while user data still exists
        const currentUser = JSON.parse(localStorage.getItem('finAI_user'));
        if (currentUser) {
          App.syncState(); // Push latest state to DB before logout
        }
        App.logout();
      }
      shell.style.display = 'none';
      document.querySelector(hash).classList.add('active');
    } else {
      // Security check: if not logged in, redirect to login!
      const user = JSON.parse(localStorage.getItem('finAI_user'));
      if (!user) {
        window.location.hash = '#login';
        return;
      }

      shell.style.display = 'flex';
      document.querySelector(hash)?.classList.add('active');
      App.updateNav(hash);

      // Render page data if available
      if (appData) App.renderPage(hash);
    }
  },

  updateNav(hash) {
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.querySelector(`.nav-item a[href="${hash}"]`)?.parentElement.classList.add('active');

    const titles = {
      '#dashboard': 'Overview Dashboard',
      '#analysis': 'Financial Analysis',
      '#projections': 'Future Projections',
      '#goals': 'Financial Goals',
      '#investments': 'Smart Investments',
      '#insights': 'AI Insights & Alerts'
    };
    document.getElementById('page-title').textContent = titles[hash] || 'Dashboard';
  },

  async triggerUpdate(isDashboard, btnElement) {
    let originalText = '';
    if (btnElement) {
      originalText = btnElement.innerHTML;
      btnElement.innerHTML = '<span class="spinner"></span> Analyzing...';
    }

    let inc, exp, sav, debt;
    if (isDashboard) {
      inc = Number(document.getElementById('dash-inp-income').value) || 0;
      exp = Number(document.getElementById('dash-inp-expenses').value) || 0;
      sav = Number(document.getElementById('dash-inp-savings').value) || 0;
      debt = Number(document.getElementById('dash-inp-debt').value) || 0;

      // Sync back to initial input form
      if (document.getElementById('inp-income')) {
        document.getElementById('inp-income').value = inc;
        document.getElementById('inp-expenses').value = exp;
        document.getElementById('inp-savings').value = sav;
        document.getElementById('inp-debt').value = debt;
      }
    } else {
      inc = Number(document.getElementById('inp-income').value) || 0;
      exp = Number(document.getElementById('inp-expenses').value) || 0;
      sav = Number(document.getElementById('inp-savings').value) || 0;
      debt = Number(document.getElementById('inp-debt').value) || 0;

      // Sync forward to dashboard input form
      const dInc = document.getElementById('dash-inp-income');
      if (dInc) {
        dInc.value = inc;
        document.getElementById('dash-inp-expenses').value = exp;
        document.getElementById('dash-inp-savings').value = sav;
        document.getElementById('dash-inp-debt').value = debt;
      }
    }
    
    // Save to localStorage
    localStorage.setItem('finAI_profile', JSON.stringify({inc, exp, sav, debt}));

    // 1. Generate frontend data structure
    appData = FinData.compute(inc, exp, sav, debt);
    
    if (userActivities === null) {
      userActivities = appData.activities;
    } else {
      appData.activities = userActivities;
    }

    // 2. Try backend AI pipeline
    try {
      const payload = FinData.toApiPayload(inc, exp, sav, debt);
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (res.ok) {
        const aiData = await res.json();
        if (!aiData.error) {
          appData.healthScore = aiData.health_score;
          appData.stress = aiData.stress_level;
          appData.personality = aiData.personality;
          appData.riskScore = aiData.risk_score;
          if (aiData.investment_allocation) appData.allocation = Object.keys(aiData.investment_allocation).map(k => ({
            name: k, pct: parseInt(aiData.investment_allocation[k]), color: k === 'Equity' ? '#3B82F6' : k === 'Debt' ? '#10B981' : '#F59E0B'
          }));

          if (aiData['6_month_projection']) {
            const proj = aiData['6_month_projection'];
            let cumSav = sav; // Start accumulating from current savings

            // Update the dashboard line chart (Profit vs Loss) using AI surplus projections
            appData.projection = proj.map((surplus_val, idx) => ({
              month: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'][idx] || `M${idx + 1}`,
              profit: inc, // Assuming income stays relatively same
              loss: inc - surplus_val // Expenses = Income - Surplus
            }));

            // Update the forecast chart (Cumulative Savings)
            proj.forEach((surplus_val, idx) => {
              cumSav += surplus_val;
              if (appData.forecast6[idx]) {
                appData.forecast6[idx].savings = cumSav;
              }
              if (appData.forecast12[idx]) {
                appData.forecast12[idx].savings = cumSav;
              }
            });
          }

          if (aiData.goal_analysis) {
            const goal = aiData.goal_analysis;
            // Update the first goal (Emergency Fund) with AI feasibility
            if (appData.goals && appData.goals[0]) {
              appData.goals[0].status = goal.achievable ? 'On Track' : 'At Risk';
              appData.goals[0].months = goal.months_needed || 'N/A';
            }
          }

          // Generate true AI Insights for the dashboard
          appData.insights = [];
          if (appData.stress === 'High') {
            appData.insights.push({ text: 'High Financial Stress Detected', sub: 'Your AI profile indicates high stress. Focus on reducing debt.', type: 'danger', icon: '🚨' });
          } else {
            appData.insights.push({ text: 'Stress Levels are Manageable', sub: 'Your AI profile indicates stable financial stress.', type: 'success', icon: '✅' });
          }

          if (appData.personality === 'Saver' || appData.personality === 'Stable') {
            appData.insights.push({ text: 'Healthy Spending Personality', sub: `Classified as ${appData.personality}. Keep up the good habits.`, type: 'success', icon: '🧠' });
          } else {
            appData.insights.push({ text: 'Risky Spending Personality', sub: `Classified as ${appData.personality}. Watch your discretionary expenses.`, type: 'warning', icon: '⚠️' });
          }

          // Generate true AI Alerts
          appData.alerts = [
            { title: 'AI Profile Updated', desc: `Health score assessed at ${appData.healthScore}.`, time: 'Just now', type: appData.healthScore > 60 ? 'success' : 'warning', icon: '⚡' },
            { title: 'Goal Feasibility', desc: appData.goals[0].status === 'On Track' ? 'You are on track to hit your main goal.' : 'Your main goal is currently at risk.', time: 'Just now', type: appData.goals[0].status === 'On Track' ? 'success' : 'danger', icon: '🎯' },
            ...appData.alerts.slice(0, 3)
          ];

        }
      }
    } catch (err) {
      console.warn('Backend /predict failed, using local simulation.', err);
    }

    await App.syncState();

    if (btnElement) {
      setTimeout(() => {
        btnElement.innerHTML = originalText;
        window.location.hash = '#dashboard';
        if (isDashboard) {
          App.renderPage('#dashboard');
        }
      }, 500);
    } else {
      // If triggered by typing, update instantly
      const hash = window.location.hash || '#dashboard';
      if (['#dashboard', '#analysis', '#projections', '#goals', '#investments', '#insights'].includes(hash)) {
        App.renderPage(hash);
      }
    }
  },

  renderPage(hash) {
    if (hash === '#dashboard') Render.dashboard();
    else if (hash === '#analysis') Render.analysis();
    else if (hash === '#projections') Render.projections();
    else if (hash === '#goals') Render.goals();
    else if (hash === '#investments') Render.investments();
    else if (hash === '#insights') Render.insights();
  },

  swapCard(index) {
    activeCardIndex = index;
    App.renderPage('#dashboard');
  },

  async contributeToGoal(index) {
    const goal = appData.goals[index];
    const amount = prompt(`How much would you like to contribute to "${goal.name}"?`);
    
    if (amount && !isNaN(amount) && Number(amount) > 0) {
      const val = Number(amount);
      
      // Update goal state
      goal.current += val;
      if (goal.current >= goal.target) {
        goal.current = goal.target;
        goal.months = 0;
      } else {
        goal.months = Math.ceil((goal.target - goal.current) / Math.max(appData.surplus, 1));
      }
      localStorage.setItem('finAI_goals', JSON.stringify(appData.goals));
      
      // Calculate new percentage
      const newPct = Math.min(100, Math.round((goal.current / goal.target) * 100));
      
      // Update UI elements immediately
      const bar = document.getElementById(`goal-bar-${index}`);
      if (bar) {
          bar.style.width = newPct + '%';
          if (newPct >= 100) {
              bar.classList.remove('gradient-fill', 'striped');
              bar.style.background = 'var(--emerald)';
              bar.parentElement.style.background = '#D1FAE5';
          }
      }
      
      const currentEl = document.getElementById(`goal-current-${index}`);
      if (currentEl) currentEl.textContent = '₹' + goal.current.toLocaleString();
      
      const pctEl = document.getElementById(`goal-pct-${index}`);
      if (pctEl) {
          pctEl.textContent = newPct + '%';
          if (newPct >= 100) pctEl.style.color = 'var(--emerald)';
      }
      
      const metaEl = document.getElementById(`goal-meta-${index}`);
      if (metaEl) {
          metaEl.textContent = goal.current >= goal.target ? 'Goal Completed 🎉' : (goal.months > 0 ? goal.months + ' months left' : 'Calculating...');
          if (newPct >= 100) {
              metaEl.style.color = 'var(--emerald)';
              metaEl.style.fontWeight = '600';
          }
      }
      
      // Update Goal Name rendering dynamically
      const titleEl = document.querySelector(`.goal-card:nth-child(${index + 1}) .goal-name`);
      if (titleEl && newPct >= 100 && !titleEl.textContent.includes('✅')) {
          titleEl.textContent = '✅ ' + goal.name;
          titleEl.style.color = 'var(--emerald)';
          titleEl.style.fontWeight = '700';
      }
      
      // Add to activities
      App.logActivity('Goal Progress', '🎯', val, 'Completed', `Funded: ${goal.name}`, true);
      Render.activitiesList('dash-activities', 5);
      Render.goals(); // refresh buttons and UI
      await App.syncState();
    }
  },
  
  async deleteGoal(index) {
    if (confirm('Are you sure you want to delete this goal?')) {
        const goal = appData.goals[index];
        appData.goals.splice(index, 1);
        localStorage.setItem('finAI_goals', JSON.stringify(appData.goals));
        App.logActivity('Goal Deleted', '🗑️', 0, 'Completed', goal.name, false);
        Render.goals();
        Render.activitiesList('dash-activities', 5);
        await App.syncState();
    }
  },

  async syncState() {
    const user = JSON.parse(localStorage.getItem('finAI_user'));
    if (!user || !user.id) return;
    
    const profile = JSON.parse(localStorage.getItem('finAI_profile'));
    const goals = appData ? appData.goals : (JSON.parse(localStorage.getItem('finAI_goals')) || []);
    const cards = userCards;
    const activities = userActivities;
    
    const statePayload = {
      user_id: user.id,
      state: {
        profile,
        goals,
        cards,
        activities
      }
    };
    
    try {
      await fetch('/sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(statePayload)
      });
    } catch (err) {
      console.error("Failed to sync state to database:", err);
    }
  },

  logout() {
    localStorage.removeItem('finAI_user');
    localStorage.removeItem('finAI_profile');
    localStorage.removeItem('finAI_goals');
    localStorage.removeItem('finAI_cards');
    localStorage.removeItem('finAI_activities');
    appData = null;
    userCards = [];
    userActivities = [];
  }
};

// ===== RENDER LOGIC =====
const Render = {
  activitiesList(containerId, limit) {
    const tbody = document.getElementById(containerId);
    if (!tbody || !appData) return;
    
    if (appData.activities.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; padding: 24px; color: var(--text-light)">No recent activities.</td></tr>';
      return;
    }

    const sym = currencySymbols[currentCurrency] || '₹';
    tbody.innerHTML = appData.activities.slice(0, limit).map(a => {
      let amountColor = 'var(--text)';
      let sign = '';
      if (a.isExpense === true) {
        amountColor = 'var(--red)';
        sign = '-';
      } else if (a.isExpense === false) {
        amountColor = 'var(--emerald)';
        sign = '+';
      } else if (['Investment', 'Transfer', 'Goal Progress'].includes(a.type)) {
        amountColor = 'var(--emerald)';
        sign = '+';
      }

      // Backward compatibility for old manual entries
      let statusText = a.status;
      let noteText = a.note;
      if (!a.note && a.status && a.status !== 'Completed' && a.status !== 'Pending' && a.status !== 'In Progress') {
        noteText = a.status;
        statusText = 'Completed';
      }

      return `
      <tr style="vertical-align: middle;">
        <td>
          <div style="display:flex; align-items:center; gap:12px; min-height:44px">
            <div class="stat-icon" style="flex-shrink:0; background:#F3F4F6; width:36px; height:36px; font-size:18px; border-radius:10px; display:flex; align-items:center; justify-content:center;">${a.icon}</div>
            <div style="display:flex; flex-direction:column; justify-content:center; text-align:left;">
              <div style="font-weight:600; color:var(--text); line-height:1.2">${a.type}</div>
              ${noteText ? `<div style="font-size:11px; color:var(--text-light); margin-top:4px; line-height:1.2; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:180px;">${noteText}</div>` : ''}
            </div>
          </div>
        </td>
        <td style="font-weight:600; color:${amountColor}">${sign}${sym}${a.amount.toLocaleString()}</td>
        <td><span class="pill ${statusText === 'Completed' ? 'pill-emerald' : statusText === 'Pending' ? 'pill-amber' : 'pill-gray'}"><span class="pill-dot"></span>${statusText}</span></td>
        <td style="color:var(--text-light); font-size:12px">${a.date}</td>
      </tr>
      `;
    }).join('');
  },

  dashboard() {
    const d = appData;

    // Handle currency conversion
    const rate = exchangeRates[currentCurrency];
    const sym = currencySymbols[currentCurrency];
    const convertedBalance = Math.round(d.balance * rate);

    document.getElementById('dash-balance').textContent = sym + convertedBalance.toLocaleString();

    // Dynamic Growth Stat (+7% this month)
    const growthPct = d.balance > 0 ? (d.surplus / d.balance) * 100 : 0;
    const statEl = document.getElementById('dash-balance-stat');
    if (statEl) {
      if (growthPct >= 0) {
        statEl.className = 'stat-trend trend-up mt-8';
        statEl.innerHTML = `▲ ${growthPct.toFixed(1)}% this month`;
      } else {
        statEl.className = 'stat-trend trend-down mt-8';
        statEl.innerHTML = `▼ ${Math.abs(growthPct).toFixed(1)}% this month`;
      }
    }

    Anim.countUp(document.getElementById('dash-score'), d.healthScore);
    document.getElementById('dash-stress').textContent = d.stress;
    document.getElementById('dash-pers').textContent = d.personality;
    document.getElementById('dash-risk').textContent = d.riskScore;

    // Dynamic insights for stats grid
    let scoreTxt = "At risk. Focus on debt reduction.";
    if (d.healthScore >= 80) scoreTxt = "Exceptional financial standing. Prime credit readiness.";
    else if (d.healthScore >= 60) scoreTxt = "Good standing. Meets standard loan criteria.";
    else if (d.healthScore >= 40) scoreTxt = "Average health. Requires optimization.";
    const scoreInsight = document.getElementById('dash-score-insight');
    if (scoreInsight) scoreInsight.textContent = scoreTxt;

    let stressTxt = "High vulnerability to economic downturns.";
    if (d.stress === 'Low') stressTxt = "Adequate liquidity. Resilient to market shocks.";
    else if (d.stress === 'Medium') stressTxt = "Moderate exposure to unexpected expenses.";
    const stressInsight = document.getElementById('dash-stress-insight');
    if (stressInsight) stressInsight.textContent = stressTxt;

    let persTxt = "Aggressive strategy. High volatility.";
    if (d.personality === 'Saver') persTxt = "High capital accumulation. Ready for investments.";
    else if (d.personality === 'Stable') persTxt = "Balanced cash flow. Consistent portfolio growth.";
    else if (d.personality === 'Impulsive') persTxt = "Erratic outflows. Impacts long-term compounding.";
    const persInsight = document.getElementById('dash-pers-insight');
    if (persInsight) persInsight.textContent = persTxt;

    let riskTxt = "Conservative. Capital preservation focused.";
    if (d.riskScore >= 80) riskTxt = "Aggressive portfolio. High market volatility.";
    else if (d.riskScore >= 50) riskTxt = "Balanced exposure to equities and debt.";
    const riskInsight = document.getElementById('dash-risk-insight');
    if (riskInsight) riskInsight.textContent = riskTxt;
    
    // Progress
    Anim.animateProgress(document.getElementById('dash-spend-bar'), d.spendingPct);
    document.getElementById('dash-spend-text').textContent = d.spendingPct + '%';

    // Charts
    Charts.lineChart('dash-chart', d.projection, { lines: [{ key: 'profit', color: '#3B82F6' }, { key: 'loss', color: '#18181B', dash: true }] });
    Charts.radialProgress('dash-health-ring', d.healthScore);
    Charts.sparkline('dash-spark', d.projection.map(p => p.profit));

    // Lists
    Render.activitiesList('dash-activities', 5);

    document.getElementById('dash-insights').innerHTML = d.insights.slice(0, 2).map(i => `
      <div class="insight-card">
        <div class="insight-icon" style="background:${i.type === 'success' ? '#D1FAE5' : i.type === 'warning' ? '#FEF3C7' : '#FEE2E2'}">${i.icon}</div>
        <div>
          <div class="insight-text">${i.text}</div>
          <div class="insight-sub">${i.sub}</div>
        </div>
      </div>
    `).join('');

    Anim.staggerIn('.card');
    Render.cards();
  },

  cards() {
    const activeContainer = document.getElementById('active-card-container');
    const cardsList = document.getElementById('cards-list');

    if (!activeContainer || !cardsList) return;

    if (userCards.length === 0) {
      activeContainer.innerHTML = `
        <div style="height: 190px; width: 100%; max-width: 320px; border: 2px dashed #D1D5DB; border-radius: 16px; display: flex; flex-direction: column; align-items: center; justify-content: center; color: var(--text-light); margin: 0 auto; background: #F9FAFB; cursor: default;">
          <div style="font-size: 32px; margin-bottom: 8px; opacity: 0.5">💳</div>
          <div style="font-size: 14px; font-weight: 600; color: var(--text)">No Active Cards</div>
          <div style="font-size: 12px; margin-top: 4px">Add a new card to manage it here.</div>
        </div>
      `;
      activeContainer.onclick = null;
      cardsList.innerHTML = '<div style="font-size:12px; color:var(--text-light); padding:16px; text-align: center; width: 100%">Your added cards will appear here.</div>';
      return;
    }

    const activeCard = userCards[activeCardIndex];
    
    let cardHTML = '';
    
    if (activeCard.type === 'mastercard') {
      cardHTML = `
        <div class="credit-card mastercard-design">
           <div class="cc-bg-curve"></div>
           <div class="flex-between" style="position:relative; z-index:2">
               <div class="cc-info" style="font-weight:400; opacity:0.9">world</div>
               <div style="font-size:16px; opacity:0.8">)))</div>
           </div>
           <div class="cc-chip"></div>
           <div class="cc-number">${activeCard.number}</div>
           <div class="cc-bottom">
               <div>
                   <div class="cc-label">Valid Thru</div>
                   <div class="cc-info">${activeCard.expiry}</div>
                   <div class="cc-info" style="margin-top:2px; font-size:11px">${activeCard.name.toUpperCase()}</div>
               </div>
               <div class="cc-circles" style="position:relative; top:0; right:0; transform:scale(0.8)"><span></span><span></span></div>
           </div>
        </div>
      `;
    } else if (activeCard.type === 'visa') {
      cardHTML = `
        <div class="credit-card visa-design">
           <div class="cc-pinstripes"></div>
           <div class="cc-info" style="font-weight:400; opacity:0.9">Visa Platinum</div>
           <div class="cc-chip" style="margin-top:10px"></div>
           <div class="cc-number" style="font-size:16px">${activeCard.number}</div>
           <div class="cc-bottom">
               <div style="max-width:60%">
                   <div class="cc-label">Card Holder</div>
                   <div class="cc-info" style="font-size:11px">${activeCard.name.toUpperCase()}</div>
               </div>
               <div style="text-align:right">
                   <div style="font-size:20px; font-weight:900; font-style:italic; line-height:1">VISA</div>
                   <div style="font-size:8px; font-weight:400; text-transform:uppercase">Platinum</div>
               </div>
           </div>
        </div>
      `;
    } else if (activeCard.type === 'chase') {
      cardHTML = `
        <div class="credit-card chase-design">
           <div class="cc-chase-lines"></div>
           <div class="flex" style="align-items:center; gap:8px; position:relative; z-index:2">
               <div style="width:20px; height:20px; background:white; clip-path:polygon(0 0, 100% 0, 100% 100%, 0 100%, 0 0, 3px 3px, 3px 17px, 17px 17px, 17px 3px, 3px 3px);"></div>
               <div style="line-height:1">
                   <div style="font-size:12px; font-weight:600; letter-spacing:1px">SAPPHIRE</div>
                   <div style="font-size:8px; font-weight:300; letter-spacing:0.5px">PREFERRED</div>
               </div>
           </div>
           <div class="cc-chip" style="margin-top:15px"></div>
           <div class="cc-number" style="font-size:16px; opacity:0.1; filter:blur(2px)">${activeCard.number}</div>
           <div class="cc-bottom">
               <div>
                   <div class="cc-label" style="opacity:0.6">${activeCard.name.toUpperCase()}</div>
               </div>
               <div style="text-align:right">
                   <div style="font-size:20px; font-weight:900; font-style:italic; line-height:1">VISA</div>
                   <div style="font-size:8px; font-weight:300; text-transform:uppercase">Signature</div>
               </div>
           </div>
        </div>
      `;
    } else if (activeCard.type === 'discover') {
      cardHTML = `
        <div class="credit-card discover-design">
           <div class="cc-discover-it">it</div>
           <div class="cc-chip"></div>
           <div style="flex:1; display:flex; align-items:center; justify-content:center; position:relative; z-index:2">
               <div style="font-size:22px; font-weight:800; letter-spacing:1px; color:#111; display:flex; align-items:center">
                   DISC<span style="display:inline-block; width:16px; height:16px; background:linear-gradient(135deg, #FF8C00, #FF4500); border-radius:50%; margin:0 2px"></span>VER
               </div>
           </div>
           <div class="cc-bottom">
               <div class="cc-info" style="color:#111; opacity:0.8">${activeCard.name.toUpperCase()}</div>
               <div style="font-size:14px; font-weight:700; color:#111; opacity:0.5">)))</div>
           </div>
        </div>
      `;
    } else if (activeCard.type === 'bofa') {
      cardHTML = `
        <div class="credit-card bofa-design">
           <div class="cc-bofa-strip">DEBIT CARD</div>
           <div class="cc-bofa-bg"></div>
           <div style="text-align:right; position:relative; z-index:2">
               <div style="font-size:12px; font-weight:700">Bank of America</div>
           </div>
           <div class="cc-chip"></div>
           <div class="cc-number" style="font-size:16px">${activeCard.number}</div>
           <div class="cc-bottom">
               <div>
                   <div class="cc-label">Valid Thru</div>
                   <div class="cc-info">${activeCard.expiry}</div>
                   <div class="cc-info" style="margin-top:2px; font-size:11px">${activeCard.name.toUpperCase()}</div>
               </div>
               <div style="text-align:right">
                   <div style="font-size:10px; font-weight:700; opacity:0.8; margin-bottom:2px">DEBIT</div>
                   <div style="font-size:18px; font-weight:900; font-style:italic; line-height:1">VISA</div>
               </div>
           </div>
        </div>
      `;
    }

    // Render Active Card
    activeContainer.innerHTML = cardHTML;

    // Render Active Card
    activeContainer.innerHTML = cardHTML;

    activeContainer.onclick = () => {
      editingCardIndex = activeCardIndex;
      document.getElementById('card-modal-title').textContent = 'Edit Card Details';
      document.getElementById('card-name').value = activeCard.name;
      document.getElementById('card-number').value = activeCard.number;
      document.getElementById('card-expiry').value = activeCard.expiry;
      const typeContainer = document.getElementById('card-type')?.parentElement;
      if (typeContainer) typeContainer.style.display = 'none';
      document.getElementById('card-modal').style.display = 'flex';
    };

    // Render Other Cards List
    const otherCards = userCards.filter((_, i) => i !== activeCardIndex);
    cardsList.innerHTML = otherCards.map((card) => {
      const originalIndex = userCards.indexOf(card);
      return `
        <div class="wallet-mini" style="cursor:pointer; min-width:140px; padding:16px; border:1px solid #eee" onclick="App.swapCard(${originalIndex})">
          <div class="flex-between mb-8">
            <span style="font-size:18px">💳</span>
            <span style="font-size:10px; font-weight:600; opacity:0.6">${card.number.slice(-4)}</span>
          </div>
          <div style="font-size:11px; font-weight:600; text-align:left; white-space:nowrap; overflow:hidden; text-overflow:ellipsis">${card.bank}</div>
        </div>
      `;
    }).join('');

    if (otherCards.length === 0) {
      cardsList.innerHTML = '<div style="font-size:12px; color:var(--text-light); padding:16px">No other cards found.</div>';
    }
  },

  analysis() {
    const d = appData;
    document.getElementById('ana-inc').textContent = '₹' + d.income.toLocaleString();
    document.getElementById('ana-exp').textContent = '₹' + d.expenses.toLocaleString();
    document.getElementById('ana-sav').textContent = Math.round(d.savingsRate * 100) + '%';
    document.getElementById('ana-debt').textContent = Math.round(d.debtRatio * 100) + '%';

    Charts.lineChart('ana-chart', d.projection);
    Charts.donutChart('ana-donut', d.expenseCategories, { centerText: 'Expenses', centerSub: 'This Month' });

    document.getElementById('ana-legend').innerHTML = d.expenseCategories.map(c => `
      <div class="donut-legend-item">
        <div class="donut-legend-left"><div class="donut-legend-color" style="background:${c.color}"></div>${c.name}</div>
        <div class="donut-legend-pct">${c.pct}%</div>
      </div>
    `).join('');
  },

  projections() {
    Charts.lineChart('proj-chart', appData.forecast12, { lines: [{ key: 'savings', color: '#10B981' }, { key: 'income', color: '#3B82F6' }] });
  },

  goals() {
    document.getElementById('goals-grid').innerHTML = appData.goals.map((g, i) => {
      const pct = Math.min(100, Math.round((g.current / g.target) * 100));
      const isComplete = g.current >= g.target;
      return `
      <div class="card goal-card">
        <div class="flex-between">
            <div class="goal-name" style="${isComplete ? 'color: var(--emerald); font-weight:700;' : ''}">${isComplete ? '✅ ' : ''}${g.name}</div>
            <div class="flex gap-8">
              ${!isComplete ? `<button class="btn btn-outline btn-sm" style="padding:4px 10px; font-size:11px" onclick="App.contributeToGoal(${i})">+ Add Funds</button>` : ''}
              <button class="btn btn-outline btn-sm" style="padding:4px 10px; font-size:11px; color:var(--red); border-color:var(--red);" onclick="App.deleteGoal(${i})">Delete</button>
            </div>
        </div>
        <div class="goal-target">Target: ₹${g.target.toLocaleString()} (Current: <span id="goal-current-${i}">₹${g.current.toLocaleString()}</span>)</div>
        <div class="progress-container" style="height:8px; ${isComplete ? 'background:#D1FAE5;' : ''}">
            <div id="goal-bar-${i}" class="progress-fill ${isComplete ? '' : 'gradient-fill striped'}" style="width:${pct}%; ${isComplete ? 'background:var(--emerald);' : ''}"></div>
        </div>
        <div class="goal-meta">
          <span id="goal-pct-${i}" style="font-weight:600; ${isComplete ? 'color:var(--emerald);' : ''}">${pct}%</span>
          <span id="goal-meta-${i}" style="color:${isComplete ? 'var(--emerald)' : 'var(--text-light)'}; font-weight:${isComplete ? '600' : 'normal'}">${isComplete ? 'Goal Completed 🎉' : (g.months > 0 ? g.months + ' months left' : 'Calculating...')}</span>
        </div>
      </div>`;
    }).join('');
    Anim.staggerIn('.goal-card');
  },

  investments() {
    Charts.donutChart('inv-donut', appData.allocation, { centerText: 'Risk:' + appData.riskScore, centerSub: 'Tolerance' });
    document.getElementById('inv-list').innerHTML = appData.allocation.map(a => `
      <div class="donut-legend-item" style="padding:16px 0">
        <div class="donut-legend-left"><div class="donut-legend-color" style="background:${a.color};width:16px;height:16px;border-radius:6px"></div><span style="font-size:15px">${a.name}</span></div>
        <div style="font-size:18px;font-weight:700">${a.pct}%</div>
      </div>
    `).join('');
  },

  insights() {
    document.getElementById('alerts-list').innerHTML = appData.alerts.map(a => `
      <div class="alert-item">
        <div class="alert-icon" style="background:${a.type === 'success' ? '#D1FAE5' : a.type === 'warning' ? '#FEF3C7' : '#FEE2E2'}">${a.icon}</div>
        <div class="alert-content">
          <div class="alert-title">${a.title}</div>
          <div class="alert-desc">${a.desc}</div>
        </div>
        <div class="alert-time">${a.time}</div>
      </div>
    `).join('');
    Anim.staggerIn('.alert-item');

    // Update behavioral summary
    const bTitle = document.getElementById('behavior-title');
    const bDesc = document.getElementById('behavior-desc');
    if (bTitle && bDesc) {
      bTitle.textContent = appData.personality;
      let desc = "";
      if (appData.personality === 'Saver') desc = "You demonstrate consistent saving habits and excellent debt management. Keep up the great work!";
      else if (appData.personality === 'Stable') desc = "Your financial behavior is stable and balanced. You maintain healthy ratios across the board.";
      else if (appData.personality === 'Impulsive') desc = "Your spending patterns indicate impulsive tendencies. Consider setting stricter budgets for discretionary spending.";
      else if (appData.personality === 'Risk-Oriented') desc = "You take on higher financial risks. Ensure you have a solid emergency fund to back up your risk tolerance.";
      bDesc.textContent = desc;
    }
  }
};

document.addEventListener('DOMContentLoaded', App.init);
