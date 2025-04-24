// Chart utility functions for time series imputation

const createTimeSeriesChart = (chartElement, dataResponse) => {
    if (!chartElement) return;

    Plotly.purge(chartElement);

    if (!dataResponse) return;

    try {
        // Process data points based on the returned data
        const isDatetime = dataResponse.index_type === 'datetime';

        // Parse data without interpolation
        const parsedData = dataResponse.data.map((point) => {
            let xVal = point.x;
            if (isDatetime && typeof xVal === 'string') {
                xVal = new Date(xVal);
            }

            return {
                x: xVal,
                y: point.y,
                missing: point.missing
            };
        });

        // Create segments for the chart
        const segments = [];
        let currentSegment = {
            x: [],
            y: [],
            missing: parsedData[0]?.missing || false,
            start: 0
        };

        parsedData.forEach((point, idx) => {
            if (point.missing !== currentSegment.missing) {
                segments.push({...currentSegment, end: idx - 1});

                currentSegment = {
                    x: [point.x],
                    y: [point.y],
                    missing: point.missing,
                    start: idx
                };
            } else {
                currentSegment.x.push(point.x);
                currentSegment.y.push(point.y);
            }
        });

        if (currentSegment.x.length > 0) {
            segments.push({...currentSegment, end: parsedData.length - 1});
        }

        // Create Plotly traces for each segment
        const data = segments.map((segment, idx) => {
            // Only create traces for non-missing segments
            if (segment.missing) {
                return null;
            }

            // segment tracking for legend
            const isFirstOfType = segments
                .slice(0, idx)
                .every(prev => prev.missing !== segment.missing);

            const trace = {
                x: segment.x,
                y: segment.y,
                type: 'scatter',
                mode: 'lines',
                showlegend: isFirstOfType,
                name: 'Available Data',
                line: {
                    color: 'rgb(0, 100, 200)',
                    width: 2
                },
                hoverinfo: 'x+y'
            };
            return trace;
        }).filter(trace => trace !== null);

        // Add a single trace for missing values in the legend
        data.push({
            x: [],
            y: [],
            type: 'scatter',
            mode: 'lines',
            showlegend: true,
            name: 'Missing Values',
            line: {
                color: 'rgba(255, 0, 0, 0.9)',
                width: 0
            },
            hoverinfo: 'skip'
        });

        // Configure the layout
        const layout = {
            title: `${dataResponse.column_name}`,
            xaxis: {
                title: isDatetime ? 'Date' :
                       dataResponse.x_column === 'index' ? 'Index' : dataResponse.x_column,
                type: isDatetime ? 'date' : null,
                tickformat: isDatetime ? '%Y-%m-%d' : null,
                tickangle: -45,
                automargin: true
            },
            yaxis: {
                title: dataResponse.column_name,
                automargin: true
            },
            hovermode: 'closest',
            legend: {
                orientation: 'h',
                yanchor: 'bottom',
                y: 1.02,
                xanchor: 'right',
                x: 1
            },
            margin: {
                l: 50,
                r: 50,
                t: 50,
                b: 80
            },
            height: 350,
            plot_bgcolor: 'rgba(240, 240, 240, 0.5)',
            paper_bgcolor: 'rgba(255, 255, 255, 0)'
        };

        const config = {
            responsive: true,
            displayModeBar: true
        };

        Plotly.newPlot(chartElement, data, layout, config);
    } catch (error) {
        console.error('Error creating chart:', error);

        // Display error message in the chart
        const errorDiv = document.createElement('div');
        errorDiv.className = 'chart-error';
        errorDiv.textContent = 'Error: Could not create chart. ' + error.message;
        chartElement.innerHTML = '';
        chartElement.appendChild(errorDiv);
    }
};

// Create a chart for imputed data showing original and imputed values
const createImputedDataChart = (chartElement, imputedData, ycolumn) => {
    if (!chartElement) return;

    Plotly.purge(chartElement);

    if (!imputedData) return;

    try {
        const originalPoints = [];
        const imputedPoints = [];
        const missingPoints = [];

        let firstMissingIndex = -1;
        let lastMissingIndex = -1;

        imputedData.data.forEach((point, index) => {
            if (point.missing) {
                if (firstMissingIndex === -1) {
                    firstMissingIndex = index;
                }
                lastMissingIndex = index;
            }
        });

        const segments = [];
        // Segment 1: Original data from start to first missing value
        if (firstMissingIndex > 0) {
            const segment1 = {
                x: [],
                y: [],
                name: 'Original Data',
                color: 'rgb(0, 100, 200)',
                missing: false
            };

            for (let i = 0; i < firstMissingIndex; i++) {
                const point = imputedData.data[i];
                if (point.y_original !== null) {
                    segment1.x.push(point.x);
                    segment1.y.push(point.y_original);
                }
            }

            if (segment1.x.length > 0) {
                segments.push(segment1);
            }
        }

        // Segment 2: Imputed data for the missing section
        if (firstMissingIndex !== -1 && lastMissingIndex !== -1) {
            const segment2 = {
                x: [],
                y: [],
                name: 'Imputed Data',
                color: 'rgb(255, 0, 0)',
                missing: true
            };

            for (let i = firstMissingIndex; i <= lastMissingIndex; i++) {
                const point = imputedData.data[i];
                if (point.missing && point.y_imputed !== null) {
                    segment2.x.push(point.x);
                    segment2.y.push(point.y_imputed);
                }
            }

            if (segment2.x.length > 0) {
                segments.push(segment2);
            }
        }

        // Segment 3: Original data from last missing value to end
        if (lastMissingIndex !== -1 && lastMissingIndex < imputedData.data.length - 1) {
            const segment3 = {
                x: [],
                y: [],
                name: 'Original Data',
                color: 'rgb(0, 100, 200)',
                missing: false
            };

            for (let i = lastMissingIndex + 1; i < imputedData.data.length; i++) {
                const point = imputedData.data[i];
                if (point.y_original !== null) {
                    segment3.x.push(point.x);
                    segment3.y.push(point.y_original);
                }
            }

            if (segment3.x.length > 0) {
                segments.push(segment3);
            }
        }

        // Create Plotly traces from segments
        const traces = segments.map((segment, index) => {
            // Only show legend for the first occurrence of each segment type
            const isFirstOfType = segments
                .slice(0, index)
                .every(prev => prev.name !== segment.name);

            return {
                x: segment.x,
                y: segment.y,
                type: 'scatter',
                mode: 'lines',
                name: segment.name,
                showlegend: isFirstOfType,
                line: {
                    color: segment.color,
                    width: 2
                },
                hoverinfo: segment.missing ? 'x+text' : 'x+y',
                hovertext: segment.missing ? Array(segment.x.length).fill(imputedPoints) : undefined
            };
        });

        // Configure the layout
        const layout = {
            title: `Imputation: ${imputedData.column_name} (${imputedData.selected_model})`,
            xaxis: {
                title: imputedData.index_type === 'datetime' ? 'Date' :
                      imputedData.x_column === 'index' ? 'Index' : imputedData.x_column,
                type: imputedData.index_type === 'datetime' ? 'date' : null,
                tickformat: imputedData.index_type === 'datetime' ? '%Y-%m-%d' : null,
                tickangle: -45,
                automargin: true
            },
            yaxis: {
                title: imputedData.column_name,
                automargin: true
            },
            hovermode: 'closest',
            legend: {
                orientation: 'h',
                yanchor: 'bottom',
                y: 1.02,
                xanchor: 'right',
                x: 1
            },
            margin: {
                l: 50,
                r: 50,
                t: 50,
                b: 80
            },
            height: 350,
            plot_bgcolor: 'rgba(240, 240, 240, 0.5)',
            paper_bgcolor: 'rgba(255, 255, 255, 0)'
        };

        const config = {
            responsive: true,
            displayModeBar: true
        };

        Plotly.newPlot(chartElement, traces, layout, config);
    } catch (error) {
        console.error('Error creating imputed chart:', error);

        // Display error message in the chart
        const errorDiv = document.createElement('div');
        errorDiv.className = 'chart-error';
        errorDiv.textContent = 'Error: Could not create imputed data chart. ' + error.message;
        chartElement.innerHTML = '';
        chartElement.appendChild(errorDiv);
    }
};

const clearChart = (chartElement) => {
    if (chartElement) {
        Plotly.purge(chartElement);
    }
};

// Export chart utilities
const TimeSeriesCharts = {
    createTimeSeriesChart,
    createImputedDataChart,
    clearChart
};