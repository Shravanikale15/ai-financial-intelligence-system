fetch("/api/financial-summary")
.then(res => res.json())
.then(data => {

    document.getElementById("income").innerText = "₹ " + data.income;
    document.getElementById("expense").innerText = "₹ " + data.expense;
    document.getElementById("balance").innerText = "₹ " + data.balance;
    document.getElementById("transactions").innerText = data.transactions;

    new Chart(document.getElementById("expenseChart"), {
        type: "doughnut",
        data: {
            labels: data.categories,
            datasets: [{
                data: data.category_amounts
            }]
        }
    });

});
