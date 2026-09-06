/**
 * Mid-Cap Stock Dashboard - Main JavaScript
 */

// Utility functions
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}

function formatPercent(value) {
    const sign = value >= 0 ? '+' : '';
    return sign + value.toFixed(2) + '%';
}

function formatNumber(value) {
    return new Intl.NumberFormat('en-US').format(value);
}

function getChangeClass(value) {
    return value >= 0 ? 'text-success' : 'text-danger';
}

function getChangeIcon(value) {
    return value >= 0 ? 'bi-arrow-up' : 'bi-arrow-down';
}

// Auto-refresh market summary every 5 minutes
function autoRefreshMarketSummary() {
    setInterval(() => {
        fetch('/api/market_summary')
            .then(response => response.json())
            .then(data => {
                // Update market summary cards if on dashboard
                for (const [name, info] of Object.entries(data)) {
                    const cards = document.querySelectorAll('.market-card');
                    cards.forEach(card => {
                        const title = card.querySelector('.card-title');
                        if (title && title.textContent === name) {
                            const priceEl = card.querySelector('h4');
                            const changeEl = card.querySelector('span');
                            if (priceEl) priceEl.textContent = info.price.toFixed(2);
                            if (changeEl) {
                                changeEl.className = info.change >= 0 ? 'text-success' : 'text-danger';
                                changeEl.innerHTML = `
                                    <i class="bi bi-arrow-${info.change >= 0 ? 'up' : 'down'}"></i>
                                    ${info.change.toFixed(2)}%
                                `;
                            }
                        }
                    });
                }
            })
            .catch(err => console.error('Error refreshing market summary:', err));
    }, 300000); // 5 minutes
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    autoRefreshMarketSummary();
    
    // Add enter key support for research symbol input
    const researchInput = document.getElementById('research-symbol');
    if (researchInput) {
        researchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                loadResearch();
            }
        });
    }
    
    // Add enter key support for quick chart symbol input
    const quickChartInput = document.getElementById('quick-chart-symbol');
    if (quickChartInput) {
        quickChartInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                loadQuickChart();
            }
        });
    }
    
    // Add enter key support for backtest symbol input
    const btSymbolInput = document.getElementById('bt-symbol');
    if (btSymbolInput) {
        btSymbolInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                runBacktest();
            }
        });
    }
});

// Toast notification system
function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container') || createToastContainer();
    
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">${message}</div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    toastContainer.appendChild(toast);
    const bsToast = new bootstrap.Toast(toast, { delay: 5000 });
    bsToast.show();
    
    toast.addEventListener('hidden.bs.toast', () => toast.remove());
}

function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
    container.style.zIndex = '11';
    document.body.appendChild(container);
    return container;
}