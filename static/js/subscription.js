// Select plan function - MUST be global for onclick handlers
function selectPlan(planName) {
    const billingToggle = document.getElementById('billingToggle');
    const billingCycle = billingToggle.checked ? 'yearly' : 'monthly';
    window.location.href = `/subscription/select?plan=${planName}&billing=${billingCycle}`;
}

// Contact sales modal functions - MUST be global for onclick handlers
function showContactSales() {
    document.getElementById('contactSalesModal').style.display = 'block';
}

function closeContactSales() {
    document.getElementById('contactSalesModal').style.display = 'none';
}

// Billing toggle and event listeners
document.addEventListener('DOMContentLoaded', function() {
    const billingToggle = document.getElementById('billingToggle');
    const monthlyPrices = document.querySelectorAll('.monthly-price');
    const yearlyPrices = document.querySelectorAll('.yearly-price');

    billingToggle.addEventListener('change', function() {
        if (this.checked) {
            monthlyPrices.forEach(p => p.style.display = 'none');
            yearlyPrices.forEach(p => p.style.display = 'inline');
        } else {
            monthlyPrices.forEach(p => p.style.display = 'inline');
            yearlyPrices.forEach(p => p.style.display = 'none');
        }
    });

    // Close modal when clicking outside
    window.onclick = function(event) {
        const modal = document.getElementById('contactSalesModal');
        if (event.target == modal) {
            modal.style.display = 'none';
        }
    }
});