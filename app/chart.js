// chart.js - Renderizador SVG Nativo interactivo con Zoom y Pan para BTC-MACHINE

let globalData = null;
let viewStartIndex = 0;
let viewEndIndex = 0;
let isDragging = false;
let dragStartX = 0;
let dragStartIndex = 0;
let dragEndIndex = 0;

async function initChart() {
    const chartContainer = document.getElementById('chart-container');
    if (!chartContainer) return;

    chartContainer.style.background = '#161820';
    chartContainer.style.minHeight = '600px';
    chartContainer.style.position = 'relative';

    // Listener para el botón de aplicar filtro
    const btnApply = document.getElementById('btn-apply-date-filter');
    if (btnApply) {
        btnApply.addEventListener('click', applyCustomDateFilter);
    }

    // Cargar datos iniciales
    await loadChartData();
    setupChartEvents();

    window.addEventListener('resize', renderSvgChart);

    // Listeners para checkboxes de EMAs
    const periods = [9, 21, 35, 50, 100, 200];
    periods.forEach(p => {
        const cb = document.getElementById(`toggle-ema${p}`);
        if (cb) {
            cb.addEventListener('change', () => {
                const paths = document.querySelectorAll(`.ema-line-${p}`);
                paths.forEach(path => {
                    path.style.display = cb.checked ? 'block' : 'none';
                });
            });
        }
    });
}

async function loadChartData(urlParams = '') {
    try {
        const response = await fetch('/api/data' + urlParams);
        const result = await response.json();

        if (!result.success) {
            alert("Error al cargar datos: " + result.message);
            return;
        }

        globalData = result;
        const klines = globalData.klines || [];
        viewStartIndex = 0;
        viewEndIndex = Math.max(0, klines.length - 1);

        renderSvgChart();
    } catch (error) {
        console.error("Error al cargar datos de la gráfica:", error);
    }
}

function highlightActivePreset(activePresetKey) {
    const buttons = document.querySelectorAll('.btn-preset');
    buttons.forEach(btn => {
        if (activePresetKey && btn.getAttribute('data-preset') === activePresetKey) {
            btn.style.background = 'rgba(59, 130, 246, 0.25)';
            btn.style.borderColor = '#3b82f6';
            btn.style.color = '#60a5fa';
            btn.style.fontWeight = 'bold';
        } else {
            btn.style.background = 'var(--surface-color-light, #2a2e3d)';
            btn.style.borderColor = 'var(--border-color, rgba(255, 255, 255, 0.1))';
            btn.style.color = '#cbd5e1';
            btn.style.fontWeight = 'normal';
        }
    });
}

function applyCustomDateFilter() {
    highlightActivePreset(null);

    const startElem = document.getElementById('filter-start-date');
    const endElem = document.getElementById('filter-end-date');

    if (!startElem || !endElem) return;

    if (!startElem.value || !endElem.value) {
        alert("Por favor selecciona ambas fechas (Desde y Hasta).");
        return;
    }

    const startTs = new Date(startElem.value).getTime();
    const endTs = new Date(endElem.value).getTime();

    if (isNaN(startTs) || isNaN(endTs)) {
        alert("Fechas inválidas seleccionadas.");
        return;
    }

    if (startTs >= endTs) {
        alert("La fecha 'Desde' debe ser anterior a 'Hasta'.");
        return;
    }

    loadChartData(`?start_ts=${startTs}&end_ts=${endTs}`);
}

function applyPresetRange(presetKey) {
    highlightActivePreset(presetKey);

    const nowMs = Date.now();
    let startTs = null;
    let endTs = nowMs;

    if (presetKey === '24h') {
        startTs = nowMs - (24 * 60 * 60 * 1000);
    } else if (presetKey === '7d') {
        startTs = nowMs - (7 * 24 * 60 * 60 * 1000);
    } else if (presetKey === '1m') {
        startTs = nowMs - (30 * 24 * 60 * 60 * 1000);
    } else if (presetKey === '6m') {
        startTs = nowMs - (180 * 24 * 60 * 60 * 1000);
    } else if (presetKey === '1y') {
        startTs = nowMs - (365 * 24 * 60 * 60 * 1000);
    } else if (presetKey === 'all') {
        startTs = nowMs - (2 * 365 * 24 * 60 * 60 * 1000);
    }

    if (startTs) {
        const startDateObj = new Date(startTs);
        const endDateObj = new Date(endTs);

        const startElem = document.getElementById('filter-start-date');
        const endElem = document.getElementById('filter-end-date');

        if (startElem) startElem.value = formatDateForInput(startDateObj);
        if (endElem) endElem.value = formatDateForInput(endDateObj);

        loadChartData(`?start_ts=${startTs}&end_ts=${endTs}`);
    }
}

function formatDateForInput(date) {
    const pad = (n) => n.toString().padStart(2, '0');
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function setupChartEvents() {
    const chartContainer = document.getElementById('chart-container');
    if (!chartContainer) return;

    // 1. Zoom con Rueda del Ratón (Wheel)
    chartContainer.addEventListener('wheel', (e) => {
        e.preventDefault();
        if (!globalData || !globalData.klines || globalData.klines.length === 0) return;

        const totalCandles = globalData.klines.length;
        const currentRange = viewEndIndex - viewStartIndex;
        const zoomFactor = e.deltaY < 0 ? 0.8 : 1.25; // Scroll Up = Zoom In, Scroll Down = Zoom Out

        let newRange = Math.round(currentRange * zoomFactor);
        newRange = Math.max(8, Math.min(totalCandles, newRange)); // Mínimo 8 velas, Máximo total de velas

        const centerIndex = Math.round((viewStartIndex + viewEndIndex) / 2);
        let newStart = centerIndex - Math.round(newRange / 2);
        let newEnd = newStart + newRange;

        if (newStart < 0) {
            newStart = 0;
            newEnd = Math.min(totalCandles - 1, newRange);
        }
        if (newEnd >= totalCandles) {
            newEnd = totalCandles - 1;
            newStart = Math.max(0, newEnd - newRange);
        }

        viewStartIndex = newStart;
        viewEndIndex = newEnd;
        renderSvgChart();
    }, { passive: false });

    // 2. Arrastre (Pan) con clic mantenido
    chartContainer.addEventListener('mousedown', (e) => {
        if (e.target.tagName === 'BUTTON' || e.target.closest('.zoom-controls')) return;
        isDragging = true;
        dragStartX = e.clientX;
        dragStartIndex = viewStartIndex;
        dragEndIndex = viewEndIndex;
        chartContainer.style.cursor = 'grabbing';
    });

    window.addEventListener('mousemove', (e) => {
        if (!isDragging || !globalData || !globalData.klines) return;
        const totalCandles = globalData.klines.length;
        const chartW = chartContainer.clientWidth || 800;
        const currentRange = dragEndIndex - dragStartIndex;
        
        const deltaX = e.clientX - dragStartX;
        const candlesShift = Math.round((deltaX / chartW) * currentRange);

        let newStart = dragStartIndex - candlesShift;
        let newEnd = dragEndIndex - candlesShift;

        if (newStart < 0) {
            newStart = 0;
            newEnd = currentRange;
        }
        if (newEnd >= totalCandles) {
            newEnd = totalCandles - 1;
            newStart = Math.max(0, newEnd - currentRange);
        }

        viewStartIndex = newStart;
        viewEndIndex = newEnd;
        renderSvgChart();
    });

    window.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            chartContainer.style.cursor = 'default';
        }
    });

    // 3. Doble Clic para reiniciar vista
    chartContainer.addEventListener('dblclick', () => {
        if (globalData && globalData.klines) {
            viewStartIndex = 0;
            viewEndIndex = globalData.klines.length - 1;
            renderSvgChart();
        }
    });
}

function zoomIn() {
    if (!globalData || !globalData.klines) return;
    const totalCandles = globalData.klines.length;
    const currentRange = viewEndIndex - viewStartIndex;
    let newRange = Math.round(currentRange * 0.75);
    newRange = Math.max(8, newRange);

    const centerIndex = Math.round((viewStartIndex + viewEndIndex) / 2);
    let newStart = centerIndex - Math.round(newRange / 2);
    let newEnd = newStart + newRange;

    if (newStart < 0) { newStart = 0; newEnd = Math.min(totalCandles - 1, newRange); }
    if (newEnd >= totalCandles) { newEnd = totalCandles - 1; newStart = Math.max(0, newEnd - newRange); }

    viewStartIndex = newStart;
    viewEndIndex = newEnd;
    renderSvgChart();
}

function zoomOut() {
    if (!globalData || !globalData.klines) return;
    const totalCandles = globalData.klines.length;
    const currentRange = viewEndIndex - viewStartIndex;
    let newRange = Math.round(currentRange * 1.3);
    newRange = Math.min(totalCandles, newRange);

    const centerIndex = Math.round((viewStartIndex + viewEndIndex) / 2);
    let newStart = centerIndex - Math.round(newRange / 2);
    let newEnd = newStart + newRange;

    if (newStart < 0) { newStart = 0; newEnd = Math.min(totalCandles - 1, newRange); }
    if (newEnd >= totalCandles) { newEnd = totalCandles - 1; newStart = Math.max(0, newEnd - newRange); }

    viewStartIndex = newStart;
    viewEndIndex = newEnd;
    renderSvgChart();
}

function resetZoom() {
    if (globalData && globalData.klines) {
        viewStartIndex = 0;
        viewEndIndex = globalData.klines.length - 1;
        renderSvgChart();
    }
}

function renderSvgChart() {
    const chartContainer = document.getElementById('chart-container');
    if (!chartContainer || !globalData) return;

    const allKlines = globalData.klines || [];
    const trades = globalData.trades || [];

    const statsCandles = document.getElementById('stats-candles');
    if (statsCandles) statsCandles.innerText = allKlines.length;

    if (allKlines.length === 0) {
        chartContainer.innerHTML = '<div style="color: #94a3b8; text-align: center; padding-top: 250px;">No hay velas disponibles para mostrar.</div>';
        return;
    }

    // Ajustar rangos válidos
    if (viewStartIndex < 0) viewStartIndex = 0;
    if (viewEndIndex >= allKlines.length) viewEndIndex = allKlines.length - 1;
    if (viewStartIndex >= viewEndIndex) viewStartIndex = Math.max(0, viewEndIndex - 8);

    const visibleKlines = allKlines.slice(viewStartIndex, viewEndIndex + 1);

    const width = chartContainer.clientWidth || (chartContainer.parentElement ? chartContainer.parentElement.clientWidth : 0) || 900;
    const height = 600;

    const paddingTop = 30;
    const paddingBottom = 40;
    const paddingLeft = 10;
    const paddingRight = 75;

    const chartW = width - paddingLeft - paddingRight;
    const chartH = height - paddingTop - paddingBottom;

    // 1. Calcular Min y Max de Precios para el rango visible
    let minPrice = Infinity;
    let maxPrice = -Infinity;

    visibleKlines.forEach(k => {
        if (k.low < minPrice) minPrice = k.low;
        if (k.high > maxPrice) maxPrice = k.high;
    });

    const priceMargin = (maxPrice - minPrice) * 0.03 || 10;
    minPrice -= priceMargin;
    maxPrice += priceMargin;

    const priceToY = (price) => {
        return paddingTop + chartH - ((price - minPrice) / (maxPrice - minPrice)) * chartH;
    };

    const numCandles = visibleKlines.length;
    const candleW = chartW / numCandles;
    const barW = Math.max(1.5, candleW * 0.7);

    const indexToX = (localIdx) => {
        return paddingLeft + localIdx * candleW + candleW / 2;
    };

    // Botones de Zoom Flotantes
    let controlsHTML = `
        <div class="zoom-controls" style="position: absolute; top: 15px; right: 85px; z-index: 10; display: flex; gap: 0.5rem; background: rgba(22, 24, 32, 0.85); padding: 0.35rem 0.6rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
            <button onclick="zoomIn()" style="background: #2a2e3d; color: #fff; border: 1px solid var(--border-color); border-radius: 4px; width: 28px; height: 28px; cursor: pointer; font-weight: bold; font-size: 1.1rem; display: flex; align-items: center; justify-content: center;" title="Acercar (Zoom In)">+</button>
            <button onclick="zoomOut()" style="background: #2a2e3d; color: #fff; border: 1px solid var(--border-color); border-radius: 4px; width: 28px; height: 28px; cursor: pointer; font-weight: bold; font-size: 1.1rem; display: flex; align-items: center; justify-content: center;" title="Alejar (Zoom Out)">-</button>
            <button onclick="resetZoom()" style="background: #2a2e3d; color: #cbd5e1; border: 1px solid var(--border-color); border-radius: 4px; padding: 0 0.5rem; height: 28px; cursor: pointer; font-size: 0.75rem; font-weight: 500;" title="Restablecer vista completa">Reset</button>
        </div>
    `;

    // Construir SVG
    let svgHTML = `<svg id="svg-chart-elem" width="100%" height="${height}" viewBox="0 0 ${width} ${height}" style="user-select: none; font-family: sans-serif; cursor: default;">`;

    // 2. Cuadrícula Horizontal (Precios)
    const priceStep = (maxPrice - minPrice) / 6;
    for (let i = 0; i <= 6; i++) {
        const pVal = minPrice + i * priceStep;
        const yPos = priceToY(pVal);
        
        svgHTML += `<line x1="${paddingLeft}" y1="${yPos}" x2="${width - paddingRight}" y2="${yPos}" stroke="rgba(255,255,255,0.06)" stroke-width="1" />`;
        svgHTML += `<text x="${width - paddingRight + 8}" y="${yPos + 4}" fill="#94a3b8" font-size="11" text-anchor="start">$${pVal.toFixed(1)}</text>`;
    }

    // 3. Cuadrícula Vertical (Fechas / Horas)
    const rangeDurationMs = (visibleKlines[numCandles - 1].time - visibleKlines[0].time) || 1;
    const timeStep = Math.max(1, Math.floor(numCandles / 7));

    for (let i = 0; i < numCandles; i += timeStep) {
        const xPos = indexToX(i);
        const k = visibleKlines[i];
        const dateStr = formatDynamicTimeLabel(k.time, rangeDurationMs);

        svgHTML += `<line x1="${xPos}" y1="${paddingTop}" x2="${xPos}" y2="${height - paddingBottom}" stroke="rgba(255,255,255,0.06)" stroke-width="1" />`;
        svgHTML += `<text x="${xPos}" y="${height - paddingBottom + 18}" fill="#94a3b8" font-size="11" text-anchor="middle">${dateStr}</text>`;
    }

    // 4. Dibujar Velas Japonesas
    visibleKlines.forEach((k, i) => {
        const x = indexToX(i);
        const yOpen = priceToY(k.open);
        const yClose = priceToY(k.close);
        const yHigh = priceToY(k.high);
        const yLow = priceToY(k.low);

        const isBullish = k.close >= k.open;
        const color = isBullish ? '#10b981' : '#ef4444';

        svgHTML += `<line x1="${x}" y1="${yHigh}" x2="${x}" y2="${yLow}" stroke="${color}" stroke-width="${Math.max(1, barW * 0.15)}" />`;

        const rectY = Math.min(yOpen, yClose);
        const rectH = Math.max(1.5, Math.abs(yClose - yOpen));
        svgHTML += `<rect x="${x - barW / 2}" y="${rectY}" width="${barW}" height="${rectH}" fill="${color}" rx="1" />`;
    });

    // 5. Dibujar EMAs
    const emaColors = {
        9: '#f59e0b',
        21: '#a855f7',
        35: '#06b6d4',
        50: '#ec4899',
        100: '#3b82f6',
        200: '#1d4ed8'
    };

    [9, 21, 35, 50, 100, 200].forEach(p => {
        const cb = document.getElementById(`toggle-ema${p}`);
        const isVisible = cb ? cb.checked : true;
        const displayStyle = isVisible ? 'block' : 'none';

        let pathD = '';
        visibleKlines.forEach((k, i) => {
            const val = k[`ema${p}`];
            if (val !== null && val !== undefined) {
                const x = indexToX(i);
                const y = priceToY(val);
                if (pathD === '') {
                    pathD += `M ${x} ${y}`;
                } else {
                    pathD += ` L ${x} ${y}`;
                }
            }
        });

        if (pathD !== '') {
            svgHTML += `<path d="${pathD}" fill="none" stroke="${emaColors[p]}" stroke-width="1.8" class="ema-line-${p}" style="display: ${displayStyle};" />`;
        }
    });

    // 6. Dibujar Trades (Flechas de Entrada -> Salida)
    const groupedTrades = {};
    trades.forEach(t => {
        if (t.trade_group !== null && t.trade_group !== undefined) {
            if (!groupedTrades[t.trade_group]) groupedTrades[t.trade_group] = [];
            groupedTrades[t.trade_group].push(t);
        }
    });

    let completedCount = 0;
    Object.keys(groupedTrades).forEach(gId => {
        const group = groupedTrades[gId];
        const buy = group.find(t => t.type === 'BUY');
        const sell = group.find(t => t.type === 'SELL');

        if (buy) {
            // Verificar si la entrada fue aprobada por el filtro ML
            const isApprovedByML = (buy.ml_approve === undefined || buy.ml_approve === 'SI' || buy.ml_approve === true || buy.ml_approve === 'true');
            if (!isApprovedByML) {
                return; // Omitir señales de entrada filtradas por Machine Learning
            }

            const buyIdx = findClosestLocalCandleIdx(visibleKlines, buy.timestamp);
            if (buyIdx !== -1) {
                const xBuy = indexToX(buyIdx);
                const yBuy = priceToY(buy.price);
                const buyDate = new Date(buy.timestamp);
                const buyTimeStr = buyDate.getHours().toString().padStart(2, '0') + ':' + buyDate.getMinutes().toString().padStart(2, '0');

                svgHTML += `<circle cx="${xBuy}" cy="${yBuy}" r="6" fill="#10b981" stroke="#ffffff" stroke-width="1.5" />`;
                svgHTML += `<text x="${xBuy}" y="${yBuy + 20}" fill="#10b981" font-size="10" font-weight="bold" text-anchor="middle">▲ Entra: $${buy.price.toFixed(1)} (${buyTimeStr})</text>`;

                if (sell) {
                    completedCount++;
                    const sellIdx = findClosestLocalCandleIdx(visibleKlines, sell.timestamp);
                    if (sellIdx !== -1) {
                        const xSell = indexToX(sellIdx);
                        const ySell = priceToY(sell.price);
                        const sellDate = new Date(sell.timestamp);
                        const sellTimeStr = sellDate.getHours().toString().padStart(2, '0') + ':' + sellDate.getMinutes().toString().padStart(2, '0');

                        const isProfit = sell.price >= buy.price;
                        const lineColor = isProfit ? '#10b981' : '#ef4444';

                        svgHTML += `<circle cx="${xSell}" cy="${ySell}" r="6" fill="#ef4444" stroke="#ffffff" stroke-width="1.5" />`;
                        svgHTML += `<text x="${xSell}" y="${ySell - 12}" fill="#ef4444" font-size="10" font-weight="bold" text-anchor="middle">▼ Sale: $${sell.price.toFixed(1)} (${sellTimeStr})</text>`;

                        svgHTML += `<line x1="${xBuy}" y1="${yBuy}" x2="${xSell}" y2="${ySell}" stroke="${lineColor}" stroke-width="2" stroke-dasharray="5 4" />`;

                        const angle = Math.atan2(ySell - yBuy, xSell - xBuy);
                        const tipX = xSell - 7 * Math.cos(angle);
                        const tipY = ySell - 7 * Math.sin(angle);
                        const arrowLen = 10;
                        const leftX = tipX - arrowLen * Math.cos(angle - Math.PI / 6);
                        const leftY = tipY - arrowLen * Math.sin(angle - Math.PI / 6);
                        const rightX = tipX - arrowLen * Math.cos(angle + Math.PI / 6);
                        const rightY = tipY - arrowLen * Math.sin(angle + Math.PI / 6);

                        svgHTML += `<polygon points="${tipX},${tipY} ${leftX},${leftY} ${rightX},${rightY}" fill="${lineColor}" />`;
                    }
                }
            }
        }
    });

    const activeInterval = globalData.interval || '5m';
    const statsInterval = document.getElementById('stats-interval');
    if (statsInterval) statsInterval.innerText = activeInterval;

    const statsTrades = document.getElementById('stats-trades');
    if (statsTrades) statsTrades.innerText = completedCount;

    svgHTML += `<text x="${paddingLeft + 10}" y="${height - paddingBottom - 10}" fill="rgba(255,255,255,0.2)" font-size="20" font-weight="bold">BTC-MACHINE (Velas ${activeInterval})</text>`;

    svgHTML += `</svg>`;

    chartContainer.innerHTML = controlsHTML + svgHTML;
}

function findClosestLocalCandleIdx(localKlines, timestamp) {
    let closestIdx = -1;
    let minDiff = Infinity;
    for (let i = 0; i < localKlines.length; i++) {
        const diff = Math.abs(localKlines[i].time - timestamp);
        if (diff < minDiff) {
            minDiff = diff;
            closestIdx = i;
        }
    }
    return closestIdx;
}

function formatDynamicTimeLabel(timestamp, rangeDurationMs) {
    const d = new Date(timestamp);
    const day = d.getDate().toString().padStart(2, '0');
    const month = (d.getMonth() + 1).toString().padStart(2, '0');
    const year = d.getFullYear().toString();
    const hours = d.getHours().toString().padStart(2, '0');
    const minutes = d.getMinutes().toString().padStart(2, '0');

    const oneDayMs = 24 * 60 * 60 * 1000;
    const sevenDaysMs = 7 * oneDayMs;

    if (rangeDurationMs > sevenDaysMs) {
        return `${day}/${month}/${year}`;
    } else if (rangeDurationMs > oneDayMs) {
        return `${day}/${month} ${hours}:${minutes}`;
    } else {
        return `${hours}:${minutes}`;
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initChart);
} else {
    initChart();
}
