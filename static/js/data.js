// ===== DATA ENGINE =====
const FinData = {
  // Generate sample data based on user input
  compute(income, expenses, savings, debt) {
    const savingsRate = income > 0 ? savings / income : 0;
    const expenseRatio = income > 0 ? expenses / income : 0;
    const surplus = income - expenses;
    const debtRatio = income > 0 ? debt / income : 0;

    // Health Score (0-100)
    let healthScore = Math.round(
      Math.max(0, Math.min(100,
        (savingsRate * 40) + ((1 - expenseRatio) * 30) + ((1 - Math.min(debtRatio, 1)) * 30)
      ) * 100) / 100
    );
    healthScore = Math.max(20, Math.min(95, Math.round(healthScore * 1.2 + 15)));

    // Stress Level
    let stress = 'Low';
    if (debtRatio > 0.5 || surplus < 0) stress = 'High';
    else if (debtRatio > 0.25 || expenseRatio > 0.7) stress = 'Medium';

    // Personality
    let personality = 'Stable';
    if (savingsRate > 0.3) personality = 'Saver';
    else if (expenseRatio > 0.85) personality = 'Impulsive';
    else if (debtRatio > 0.4) personality = 'Risk-Oriented';

    // Risk Score
    const stressNum = { Low: 0, Medium: 50, High: 100 }[stress];
    const persNum = { Saver: 30, Stable: 50, 'Risk-Oriented': 80, Impulsive: 70 }[personality];
    const riskScore = Math.round(0.5 * healthScore + 0.3 * persNum - 0.2 * stressNum);

    // Balance
    const balance = savings;

    // Monthly projection data
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug'];
    const projection = months.map((m, i) => ({
      month: m,
      profit: Math.round(income * (0.8 + Math.random() * 0.5) * (1 + i * 0.04)),
      loss: Math.round(expenses * (0.7 + Math.random() * 0.4) * (1 + i * 0.02))
    }));

    // 6/12 month forecast
    const forecast6 = [];
    const forecast12 = [];
    let cumSavings = savings;
    for (let i = 0; i < 12; i++) {
      cumSavings += surplus * (0.95 + Math.random() * 0.1);
      const point = { month: i + 1, label: `M${i + 1}`, savings: Math.round(cumSavings), income: Math.round(income * (1 + i * 0.01)), expenses: Math.round(expenses * (1 + i * 0.005)) };
      if (i < 6) forecast6.push(point);
      forecast12.push(point);
    }

    // Expense breakdown
    const expCats = [
      { name: 'Housing', pct: 35, color: '#3B82F6' },
      { name: 'Food', pct: 20, color: '#FF6B35' },
      { name: 'Transport', pct: 15, color: '#10B981' },
      { name: 'Shopping', pct: 12, color: '#F59E0B' },
      { name: 'Bills', pct: 10, color: '#8B5CF6' },
      { name: 'Other', pct: 8, color: '#6B7280' }
    ];

    // Investment allocation
    let allocation;
    if (riskScore < 40) allocation = [{ name: 'Equity', pct: 20, color: '#3B82F6' }, { name: 'Debt', pct: 60, color: '#10B981' }, { name: 'Gold', pct: 20, color: '#F59E0B' }];
    else if (riskScore < 70) allocation = [{ name: 'Equity', pct: 40, color: '#3B82F6' }, { name: 'Debt', pct: 40, color: '#10B981' }, { name: 'Gold', pct: 20, color: '#F59E0B' }];
    else allocation = [{ name: 'Equity', pct: 70, color: '#3B82F6' }, { name: 'Debt', pct: 20, color: '#10B981' }, { name: 'Gold', pct: 10, color: '#F59E0B' }];

    // Goals
    let goals = JSON.parse(localStorage.getItem('finAI_goals'));
    if (!goals || goals.length === 0) {
      goals = [
        { id: 1, name: 'Emergency Fund', target: income * 6, current: savings * 2 },
        { id: 2, name: 'Vacation Fund', target: 50000, current: 32000 },
        { id: 3, name: 'Debt Freedom', target: debt, current: debt * 0.4 },
        { id: 4, name: 'Investment Portfolio', target: 500000, current: 180000 }
      ];
    }
    
    // Recalculate months based on current surplus
    goals.forEach(g => {
        if (g.current >= g.target) {
            g.status = 'Completed';
            g.months = 0;
            g.current = g.target;
        } else {
            g.status = 'On Track';
            const remaining = g.target - g.current;
            g.months = Math.ceil(remaining / Math.max(surplus * 0.2, 1));
        }
    });

    // Recent activities (start empty)
    const activities = [];

    // Insights
    const insights = [];
    if (expenseRatio > 0.7) insights.push({ text: 'Expenses are high relative to income', sub: `${Math.round(expenseRatio * 100)}% of income goes to expenses`, type: 'warning', icon: '⚠️' });
    else insights.push({ text: 'Spending is well controlled', sub: `Only ${Math.round(expenseRatio * 100)}% of income spent`, type: 'success', icon: '✅' });

    if (savingsRate > 0.2) insights.push({ text: 'Savings rate is strong', sub: `Saving ${Math.round(savingsRate * 100)}% of income`, type: 'success', icon: '💰' });
    else insights.push({ text: 'Savings rate needs improvement', sub: `Only ${Math.round(savingsRate * 100)}% saved`, type: 'warning', icon: '📉' });

    if (debtRatio > 0.3) insights.push({ text: 'Debt ratio is slightly high', sub: `Debt is ${Math.round(debtRatio * 100)}% of income`, type: 'danger', icon: '🔴' });
    else insights.push({ text: 'Debt levels are manageable', sub: `Debt is ${Math.round(debtRatio * 100)}% of income`, type: 'success', icon: '✅' });

    insights.push({ text: surplus > 0 ? 'Monthly surplus is positive' : 'Monthly deficit detected', sub: `₹${Math.abs(surplus).toLocaleString()} ${surplus > 0 ? 'surplus' : 'deficit'}`, type: surplus > 0 ? 'success' : 'danger', icon: surplus > 0 ? '📊' : '🚨' });

    // Alerts
    const alerts = [
      { title: 'Spending Spike Detected', desc: 'Shopping expenses increased by 18% this week', time: '2h ago', type: 'warning', icon: '⚡' },
      { title: surplus > 0 ? 'Savings Growing' : 'Savings Decreasing', desc: surplus > 0 ? 'Your savings grew by 5% this month' : 'Your savings dropped — review expenses', time: '1d ago', type: surplus > 0 ? 'success' : 'danger', icon: surplus > 0 ? '📈' : '📉' },
      { title: 'Bill Payment Due', desc: 'Electricity bill of ₹2,340 due in 3 days', time: '3d ago', type: 'info', icon: '📋' },
      { title: 'Investment Opportunity', desc: 'Market conditions favor equity investment', time: '5d ago', type: 'info', icon: '💡' },
      { title: 'Goal Update', desc: 'Emergency fund is 60% complete', time: '1w ago', type: 'success', icon: '🎯' }
    ];

    return {
      income, expenses, savings, debt, surplus, balance,
      savingsRate, expenseRatio, debtRatio,
      healthScore, stress, personality, riskScore,
      projection, forecast6, forecast12,
      expenseCategories: expCats, allocation,
      goals, activities, insights, alerts,
      spendingPct: Math.min(100, Math.round(expenseRatio * 100))
    };
  },

  // Build API payload for Flask backend
  toApiPayload(income, expenses, savings, debt) {
    const savingsRate = income > 0 ? savings / income : 0;
    const expenseRatio = income > 0 ? expenses / income : 0;
    const surplus = income - expenses;
    return {
      savings_rate: savingsRate,
      expense_ratio: expenseRatio,
      income_variance: 0.05,
      expense_volatility: 0.08,
      monthly_surplus: surplus,
      expense_spike: 0,
      food_ratio: 0.2,
      entertainment_ratio: 0.1,
      discretionary_ratio: 0.15,
      goal_amount: income * 6
    };
  }
};
