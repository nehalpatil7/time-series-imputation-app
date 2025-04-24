// Main App component for Time Series Imputation
// Note: This file uses Babel JSX syntax

const { useState, useEffect, useRef } = React;
const { createTimeSeriesChart, createImputedDataChart, clearChart } = TimeSeriesCharts;
const api = TimeSeriesApi;

const candidateModels = [
    { value: '', label: 'Auto (recommended)' },
    { value: 'linear_interpolation', label: 'Linear Interpolation' },
    { value: 'spline_interpolation', label: 'Spline Interpolation' },
    { value: 'mean_imputation', label: 'Mean Imputation' },
    { value: 'knn_imputation', label: 'KNN Imputation' },
    { value: 'regression_imputation', label: 'Regression Imputation' },
    { value: 'mice_imputation', label: 'MICE Imputation' },
    { value: 'arima_imputation', label: 'ARIMA Imputation' },
    { value: 'gb_imputation', label: 'Gradient Boosting Imputation' },
    { value: 'lstm_imputation', label: 'LSTM Imputation' }
];

// ARIMA parameter defaults
const defaultArimaParams = {
    p: '',
    d: '',
    q: '',
    P: '',
    D: '',
    Q: '',
    s: '',
    y_column: ''
};

const App = () => {
    // State variables
    const [file, setFile] = useState(null);
    const [fileName, setFileName] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [datasetId, setDatasetId] = useState('');
    const [features, setFeatures] = useState(null);
    const [selectedModel, setSelectedModel] = useState('');
    const [hasTrend, setHasTrend] = useState('auto');
    const [hasSeasonality, setHasSeasonality] = useState('auto');
    const [overrideModel, setOverrideModel] = useState('');
    const [analysisResult, setAnalysisResult] = useState(null);
    const [isDragging, setIsDragging] = useState(false);
    const [columns, setColumns] = useState([]);
    const [columnTypes, setColumnTypes] = useState({});
    const [selectedXColumn, setSelectedXColumn] = useState('');
    const [selectedYColumn, setSelectedYColumn] = useState('');
    const [hasDatetimeIndex, setHasDatetimeIndex] = useState(false);
    const [indexName, setIndexName] = useState('Index');
    const [hasDateColumn, setHasDateColumn] = useState(false);
    const [dateColumnMissingError, setDateColumnMissingError] = useState(false);
    const [imputedData, setImputedData] = useState(null);
    const [isImputed, setIsImputed] = useState(false);
    // New state for ARIMA parameters
    const [arimaParams, setArimaParams] = useState({ ...defaultArimaParams, y_column: selectedYColumn });
    const [showArimaParams, setShowArimaParams] = useState(false);

    // New state for ARIMA diagnostic plots
    const [arimaDiagnostics, setArimaDiagnostics] = useState(null);
    const [loadingDiagnostics, setLoadingDiagnostics] = useState(false);

    // New state to control showing override options
    const [showingPlot, setShowingPlot] = useState(false);

    // New state for data preview
    const [dataPreview, setDataPreview] = useState(null);

    const chartRef = useRef(null);
    const imputedChartRef = useRef(null);
    const fileInputRef = useRef(null);

    // Effect to load diagnostics when ARIMA is selected
    useEffect(() => {
        if (overrideModel === 'arima_imputation' || analysisResult?.selectedModel === 'arima_imputation') {
            loadArimaDiagnostics();
        } else {
            setArimaDiagnostics(null);
        }
    }, [overrideModel, analysisResult?.selectedModel, datasetId, selectedYColumn]);

    useEffect(() => {
        if (datasetId && features) {
            setIsLoading(false);
        }
    }, [datasetId, features]);

    useEffect(() => {
        if (datasetId && features && !isLoading) {
            console.log("Redrawing chart due to dependency changes");
            createChart(features, datasetId);
        }
    }, [datasetId, hasDateColumn, dateColumnMissingError]);

    // Effect to update ARIMA parameters when selectedYColumn changes
    useEffect(() => {
        if (selectedYColumn) {
            setArimaParams(prev => ({ ...prev, y_column: selectedYColumn }));
        }
    }, [selectedYColumn]);

    // Handle ARIMA parameter changes
    const handleArimaParamChange = (param, value) => {
        if (['p', 'd', 'q', 'P', 'D', 'Q', 's'].includes(param)) {
            if (value === '') {
                setArimaParams({
                    ...arimaParams,
                    [param]: value
                });
            } else {
                setArimaParams({
                    ...arimaParams,
                    [param]: parseInt(value, 10) || 0
                });
            }
        } else {
            setArimaParams({
                ...arimaParams,
                [param]: value
            });
        }
    };

    const handleFileSelect = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile) {
            validateAndSetFile(selectedFile);
        }
    };

    const validateAndSetFile = (selectedFile) => {
        const fileExt = selectedFile.name.split('.').pop().toLowerCase();
        if (fileExt !== 'csv') {
            setError('Please upload a CSV file only');
            setFile(null);
            setFileName('');
            return;
        }

        setFile(selectedFile);
        setFileName(selectedFile.name);
        setError('');
        setFeatures(null);
        setDatasetId('');
        setAnalysisResult(null);
        setImputedData(null);
        setIsImputed(false);

        clearChart(chartRef.current);
        clearChart(imputedChartRef.current);
    };

    const uploadFile = async () => {
        if (!file) return;

        setIsLoading(true);
        setError('');

        try {
            const uploadResponse = await api.uploadDataset(file);
            const newDatasetId = uploadResponse.dataset_id;
            setDatasetId(newDatasetId);

            const preprocessResponse = await api.preprocessDataset(newDatasetId);
            setFeatures(preprocessResponse.features);

            await fetchColumnInfo(newDatasetId);

            setTimeout(() => {
                createChart(preprocessResponse.features, newDatasetId);
                setIsLoading(false);
            }, 200);
        } catch (err) {
            console.error('Error:', err);
            setError(err.response?.data?.detail || 'An error occurred while processing the file');
            setIsLoading(false);
        }
    };

    const fetchColumnInfo = async (dsId) => {
        try {
            const columnData = await api.getColumnInfo(dsId);

            setColumns(columnData.columns);
            setColumnTypes(columnData.column_types);
            setHasDatetimeIndex(columnData.has_datetime_index);
            setIndexName(columnData.index_name);

            const dateColumnNames = ['Date', 'DATE', 'date', 'datetime', 'Datetime', 'DATETIME', 'time', 'Time', 'TIME'];
            const dateColumnRegex = /^(date|time)/i;

            const hasDateCol = dateColumnNames.some(name => columnData.columns.includes(name)) ||
                columnData.columns.some(col => dateColumnRegex.test(col)) ||
                columnData.has_datetime_index;
            setHasDateColumn(hasDateCol);

            if (!hasDateCol) {
                setDateColumnMissingError(true);
                console.error("Dataset does not have a 'Date' column");
            } else {
                setDateColumnMissingError(false);
                setSelectedXColumn('Date');
            }

            // Set default Y column to first numeric column that is not the Date column
            const numericColumns = columnData.columns.filter(
                col => columnData.column_types[col] === 'numeric' && col !== 'Date'
            );
            if (numericColumns.length > 0) {
                setSelectedYColumn(numericColumns[0]);
            }
        } catch (err) {
            console.error('Error fetching column information:', err);
            setError('Failed to load column information. Please try again.');
        }
    };

    const createChart = async (features, chartDatasetId, xColumn = null, yColumn = null) => {
        const currentDatasetId = chartDatasetId || datasetId;

        if (!currentDatasetId) {
            console.error('No dataset ID available for chart creation');
            return;
        }

        try {
            if (chartRef.current) {
                clearChart(chartRef.current);

                if (dateColumnMissingError) {
                    const errorDiv = document.createElement('div');
                    errorDiv.className = 'chart-error';
                    errorDiv.textContent = 'Error: Dataset does not have a "Date" column';
                    chartRef.current.innerHTML = '';
                    chartRef.current.appendChild(errorDiv);
                    return;
                }

                // Get the dataset for visualization
                const params = {};

                if (hasDateColumn) {
                    params.x_column = 'Date';
                } else if (xColumn) {
                    params.x_column = xColumn;
                } else if (selectedXColumn) {
                    params.x_column = selectedXColumn;
                }

                if (yColumn) {
                    params.y_column = yColumn;
                } else if (selectedYColumn) {
                    params.y_column = selectedYColumn;
                }

                // Fetch data and create the chart
                const dataResponse = await api.getDatasetForChart(
                    currentDatasetId,
                    params.x_column,
                    params.y_column
                );

                createTimeSeriesChart(chartRef.current, dataResponse);

                // Get data preview - We'll extract preview data from the full dataset response
                try {
                    // Fetch the dataset preview directly from API
                    const previewData = await api.getDatasetPreview(currentDatasetId);
                    setDataPreview(previewData);
                } catch (previewErr) {
                    console.error('Error fetching data preview:', previewErr);
                    setDataPreview(null);
                }

                // Mark that the initial plot is shown
                setShowingPlot(true);
            }
        } catch (err) {
            console.error('Error creating chart:', err);
            setError('Error creating visualization. Please try again.');

            // Display error message in the chart
            if (chartRef.current) {
                const errorDiv = document.createElement('div');
                errorDiv.className = 'chart-error';
                errorDiv.textContent = 'Error: Could not create chart. ' + err.message;
                chartRef.current.innerHTML = '';
                chartRef.current.appendChild(errorDiv);
            }
        }
    };

    const analyzeData = async () => {
        if (!datasetId) return;

        setIsLoading(true);
        setError('');
        // Reset imputation state
        setImputedData(null);
        setIsImputed(false);

        try {
            // Prepare the modified features based on user inputs
            const modifiedFeatures = { ...features };

            if (hasTrend !== 'auto') {
                modifiedFeatures.has_trend = hasTrend === 'yes' ? 1 : 0;
            }

            if (hasSeasonality !== 'auto') {
                modifiedFeatures.has_seasonality = hasSeasonality === 'yes' ? 1 : 0;
            }

            // Add ARIMA parameters if manually configured
            if (showArimaParams) {
                modifiedFeatures.arima_params = {
                    order: [arimaParams.p, arimaParams.d, arimaParams.q],
                    seasonal_order: [arimaParams.P, arimaParams.D, arimaParams.Q, arimaParams.s],
                    trend: arimaParams.trend,
                    y_column: arimaParams.y_column || selectedYColumn
                };
            }

            // Request model selection
            const selectModelResponse = await api.selectModel(datasetId, overrideModel);
            const selectedModel = selectModelResponse.selected_model;
            setSelectedModel(selectedModel);

            // Update showArimaParams based on the selected model
            setShowArimaParams(selectedModel === 'arima_imputation' || overrideModel === 'arima_imputation');

            // Set analysis result with selected model & metrics from model selection
            setAnalysisResult({
                selectedModel: selectedModel,
                metrics: selectModelResponse.metrics || { note: "Click 'Impute' to run the model and generate metrics" }
            });

            setIsLoading(false);
        } catch (err) {
            console.error('Error:', err);
            setError(err.response?.data?.detail || 'An error occurred during analysis');
            setIsLoading(false);
        }
    };

    const displayImputedData = async (datasetId, yColumn) => {
        if (!datasetId || !yColumn || !selectedModel) return;

        setIsLoading(true);
        setError('');

        try {
            // Step 1: Call the impute endpoint to actually perform imputation
            // Pass ARIMA parameters if the model is ARIMA
            const imputeResponse = await api.imputeDataset(
                datasetId,
                selectedModel,
                selectedYColumn,
                selectedModel === 'arima_imputation' ? { ...arimaParams, y_column: arimaParams.y_column || selectedYColumn, hasSeasonality: hasSeasonality } : null,
            );

            // Update the analysis results with actual metrics from imputation
            setAnalysisResult({
                selectedModel: selectedModel,
                metrics: imputeResponse.evaluation_metrics || { note: "Click 'Impute' to run the model and generate metrics" }
            });

            // Step 2: Get the imputed dataset for visualization
            const imputedDataResponse = await api.getImputedDataset(datasetId, selectedModel, selectedYColumn, selectedModel === 'arima_imputation' ? { ...arimaParams, y_column: arimaParams.y_column || selectedYColumn } : null);

            // Store the imputed data in state
            setImputedData(imputedDataResponse);

            // Display imputed data on the dedicated imputed chart
            createImputedDataChart(imputedChartRef.current, imputedDataResponse, yColumn);

            // Set flag indicating imputation is complete
            setIsImputed(true);

            setIsLoading(false);
        } catch (err) {
            console.error('Error:', err);
            setError(err.response?.data?.detail || 'An error occurred while performing imputation');
            setIsLoading(false);
        }
    };

    // Load ARIMA diagnostic plots
    const loadArimaDiagnostics = async () => {
        if (!datasetId || !selectedYColumn) return;

        setLoadingDiagnostics(true);
        try {
            const diagnostics = await api.getArimaDiagnostics(datasetId, selectedYColumn);
            setArimaDiagnostics(diagnostics);
        } catch (error) {
            console.error('Error loading ARIMA diagnostics:', error);
            setError('Failed to load ARIMA diagnostic plots');
        } finally {
            setLoadingDiagnostics(false);
        }
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = () => {
        setIsDragging(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            validateAndSetFile(e.dataTransfer.files[0]);
        }
    };

    // Check if ARIMA parameters are valid
    const areArimaParamsValid = () => {
        if (overrideModel === 'arima_imputation' || (analysisResult?.selectedModel === 'arima_imputation' && !overrideModel)) {
            const requiredParams = ['p', 'd', 'q', 'P', 'D', 'Q', 's'];
            return requiredParams.every(param => arimaParams[param] !== '');
        }
        return true;
    };

    return (
        <div className="container">
            <div className="header">
                <h1>Time Series Imputation</h1>
                <p>Upload your time series data to analyze and impute missing values</p>
            </div>

            <div className="card">
                <div
                    className={`upload-container ${isDragging ? 'drag-over' : ''}`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                >
                    <input
                        type="file"
                        className="file-input"
                        onChange={handleFileSelect}
                        accept=".csv"
                        ref={fileInputRef}
                    />
                    <p>Drag and drop your CSV file here or click the button below</p>
                    <button
                        className="upload-btn"
                        onClick={() => fileInputRef.current.click()}
                    >
                        Select File
                    </button>
                    <p className="feature-note"><small><sup className="warning-sup">*</sup> The first column should be the date/time column with column name 'Date' or 'date'. <br></br><sup className="warning-sup">*</sup>The column name should not contain any special characters, only alphanumeric characters and underscores.</small></p>
                    {fileName && <p style={{ textAlign: 'center', marginTop: '10px' }}>Selected file: {fileName}</p>}
                    {error && <p className="error">{error}</p>}
                </div>

                {file && !datasetId && !isLoading && (
                    <button
                        className="analyze-btn"
                        onClick={uploadFile}
                    >
                        Upload
                    </button>
                )}

                {isLoading && <div className="spinner"></div>}
            </div>

            {features && (
                <div className="card">
                    <h3>
                        Dataset Features
                        <sup className="warning-sup">*</sup>
                    </h3>
                    <p className="feature-note"><small>* Feature extraction is an estimation. Please verify the detected features match your data.<br />* The missing rate is decided based on NaN values.</small></p>
                    <div className="feature-container">
                        <span className="feature-badge">
                            Length: {features.length} records
                        </span>
                        <span className="feature-badge">
                            Missing Rate: {(features.missing_rate * 100).toFixed(2)}%
                        </span>
                        <span className="feature-badge">
                            Has Trend: {features.has_trend ? 'Yes' : 'No'}
                        </span>
                        <span className="feature-badge">
                            Has Seasonality: {features.has_seasonality ? 'Yes' : 'No'}
                        </span>
                    </div>

                    <div className="axis-selectors">
                        <div className="control-group">
                            <label htmlFor="x-axis">X-Axis:</label>
                            <select
                                id="x-axis"
                                value="Date"
                                disabled={true}
                            >
                                <option value="Date">Date (Date/Time)</option>
                            </select>
                            {!hasDateColumn && (
                                <p className="error-message">
                                    No 'Date' column found in dataset. X-axis will use index.
                                </p>
                            )}
                        </div>

                        <div className="control-group">
                            <label htmlFor="y-axis">Y-Axis:</label>
                            <select
                                id="y-axis"
                                value={selectedYColumn}
                                onChange={(e) => {
                                    const newColumn = e.target.value;
                                    setSelectedYColumn(newColumn);
                                    createChart(features, datasetId, undefined, newColumn);
                                }}
                                disabled={columns.filter(col => columnTypes[col] === 'numeric' && col !== 'Date').length <= 1}
                            >
                                <option value="">Select Column</option>
                                {columns
                                    .filter(col => columnTypes[col] === 'numeric' && col !== 'Date')
                                    .map(col => (
                                        <option
                                            key={col}
                                            value={col}
                                        >
                                            {col} (Number)
                                        </option>
                                    ))
                                }
                            </select>
                        </div>
                    </div>

                    <div className="chart-container">
                        <div ref={chartRef} style={{ width: '100%', height: '100%' }}></div>
                    </div>

                    {/* Data Preview Table */}
                    {dataPreview && (
                        <div className="data-preview" style={{ marginTop: '20px', marginBottom: '20px' }}>
                            <h4>Data Preview</h4>
                            <div className="table-responsive">
                                <table className="data-table">
                                    <thead>
                                        <tr>
                                            {dataPreview.columns.map(column => (
                                                <th key={column}>{column}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {dataPreview.data.map((row, rowIndex) => (
                                            <tr key={rowIndex}>
                                                {dataPreview.columns.map(column => (
                                                    <td key={`${rowIndex}-${column}`}>
                                                        {row[column] === null || row[column] === undefined ?
                                                            <span className="null-value">NaN</span> :
                                                            String(row[column])}
                                                    </td>
                                                ))}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}

                    <button
                        className="analyze-btn"
                        onClick={analyzeData}
                        disabled={isLoading || !hasDateColumn || (features && features.missing_rate === 0)}
                        title={features && features.missing_rate === 0 ? "No missing values to impute in this dataset" : (hasDateColumn ? "Analyze dataset for imputation" : "Analysis requires a 'Date' column")}
                    >
                        Analyze
                    </button>
                    {!hasDateColumn && (
                        <p className="error-message" style={{ textAlign: 'center', marginTop: '10px' }}>
                            Analysis requires a "Date" column in the dataset
                        </p>
                    )}
                    {features && features.missing_rate === 0 && (
                        <p className="info-message" style={{ textAlign: 'center', marginTop: '10px' }}>
                            This dataset has no missing values. No imputation needed.
                        </p>
                    )}
                </div>
            )}

            {analysisResult && (
                <div className="card results">
                    <h3>Analysis Results</h3>
                    <p><strong>Selected Model:</strong> {analysisResult.selectedModel}</p>
                    <h4>Selected Model Information:</h4>
                    <div className="feature-container">
                        {Object.entries(analysisResult.metrics).map(([key, value]) => (
                            <span className="feature-badge" key={key}>
                                {key}: {typeof value === 'number' ? value.toFixed(4) : value}
                            </span>
                        ))}
                    </div>

                    {/* Model override options - now placed between model info and imputation */}
                    <div className="model-override-section" style={{ marginTop: '20px', marginBottom: '20px' }}>
                        <h4>Model Override Options:</h4>
                        <div className="controls">
                            <div className="control-group">
                                <label htmlFor="model">Select Model:</label>
                                <select
                                    id="model"
                                    value={overrideModel}
                                    onChange={(e) => {
                                        const newModel = e.target.value;
                                        setOverrideModel(newModel);
                                        // Show ARIMA params form if ARIMA is selected
                                        setShowArimaParams(newModel === 'arima_imputation' || analysisResult?.selectedModel === 'arima_imputation');
                                    }}
                                >
                                    {candidateModels.map(model => (
                                        <option key={model.value} value={model.value}>
                                            {model.label}
                                        </option>
                                    ))}
                                </select>
                            </div>
                        </div>

                        {/* Only show trend and seasonality options for ARIMA */}
                        {((showArimaParams || analysisResult?.selectedModel === 'arima_imputation') && !overrideModel) || overrideModel === 'arima_imputation' ? (
                            <>
                                {/* ARIMA Diagnostic Plots */}
                                <div className="arima-diagnostics" style={{ marginTop: '15px', marginBottom: '15px' }}>
                                    <h4>ARIMA Diagnostic Plots</h4>
                                    <p className="feature-note">These plots help determine the appropriate ARIMA parameters (p,d,q)</p>

                                    {loadingDiagnostics && <div className="spinner"></div>}

                                    {arimaDiagnostics && !loadingDiagnostics && (
                                        <div className="diagnostic-plots">
                                            {arimaDiagnostics.error && (
                                                <p className="error-message">{arimaDiagnostics.error}</p>
                                            )}

                                            {/* BoxCox Plot */}
                                            {arimaDiagnostics.boxcox_plot && (
                                                <div className="plot-container">
                                                    <h5>BoxCox Transformation</h5>
                                                    <img
                                                        src={`data:image/png;base64,${arimaDiagnostics.boxcox_plot}`}
                                                        alt="BoxCox Transformation"
                                                        style={{ width: '100%', maxWidth: '700px' }}
                                                    />
                                                </div>
                                            )}

                                            {/* Differenced Plot */}
                                            {arimaDiagnostics.diff_plot && (
                                                <div className="plot-container">
                                                    <h5>Differenced Data (d=1)</h5>
                                                    <img
                                                        src={`data:image/png;base64,${arimaDiagnostics.diff_plot}`}
                                                        alt="Differenced Data"
                                                        style={{ width: '100%', maxWidth: '700px' }}
                                                    />
                                                </div>
                                            )}

                                            {/* ACF Plot */}
                                            {arimaDiagnostics.acf_plot && (
                                                <div className="plot-container">
                                                    <h5>Autocorrelation Function (ACF) - helps determine q</h5>
                                                    <img
                                                        src={`data:image/png;base64,${arimaDiagnostics.acf_plot}`}
                                                        alt="ACF Plot"
                                                        style={{ width: '100%', maxWidth: '700px' }}
                                                    />
                                                </div>
                                            )}

                                            {/* PACF Plot */}
                                            {arimaDiagnostics.pacf_plot && (
                                                <div className="plot-container">
                                                    <h5>Partial Autocorrelation Function (PACF) - helps determine p</h5>
                                                    <img
                                                        src={`data:image/png;base64,${arimaDiagnostics.pacf_plot}`}
                                                        alt="PACF Plot"
                                                        style={{ width: '100%', maxWidth: '700px' }}
                                                    />
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>

                                <div className="arima-specific-controls" style={{ marginTop: '15px' }}>
                                    <h4>ARIMA Specific Options:</h4>
                                    <div className="controls">
                                        <div className="control-group">
                                            <label htmlFor="trend">Trend Override:</label>
                                            <select
                                                id="trend"
                                                value={hasTrend}
                                                onChange={(e) => setHasTrend(e.target.value)}
                                            >
                                                <option value="auto">Auto Detect</option>
                                                <option value="yes">Yes</option>
                                                <option value="no">No</option>
                                            </select>
                                        </div>

                                        <div className="control-group">
                                            <label htmlFor="seasonality">Seasonality Override:</label>
                                            <select
                                                id="seasonality"
                                                value={hasSeasonality}
                                                onChange={(e) => setHasSeasonality(e.target.value)}
                                            >
                                                <option value="auto">Auto Detect</option>
                                                <option value="yes">Yes</option>
                                                <option value="no">No</option>
                                            </select>
                                        </div>
                                    </div>
                                </div>
                            </>
                        ) : null}

                        {/* ARIMA parameters section */}
                        {showArimaParams && (
                            <div className="arima-params" style={{ marginTop: '15px' }}>
                                <h4>ARIMA Parameters</h4>
                                <p className="feature-note">Customize the ARIMA model parameters (usually p & q should be selected as most significant lag value (matching most to both ACF & PACF), d should be 1, P & Q should be selected as most significant seasonal lag value, if seasonality present, else 0)</p>

                                <div className="controls">
                                    <div className="control-group">
                                        <label htmlFor="arima-p">p (AR Order) <span className="required-asterisk">*</span></label>
                                        <input
                                            type="number"
                                            id="arima-p"
                                            min="0"
                                            max="5"
                                            value={arimaParams.p}
                                            onChange={(e) => handleArimaParamChange('p', e.target.value)}
                                            required
                                            placeholder="Enter p value"
                                        />
                                    </div>
                                    <div className="control-group">
                                        <label htmlFor="arima-d">d (Differencing) <span className="required-asterisk">*</span></label>
                                        <input
                                            type="number"
                                            id="arima-d"
                                            min="0"
                                            max="2"
                                            value={arimaParams.d}
                                            onChange={(e) => handleArimaParamChange('d', e.target.value)}
                                            required
                                            placeholder="Enter d value"
                                        />
                                    </div>
                                    <div className="control-group">
                                        <label htmlFor="arima-q">q (MA Order) <span className="required-asterisk">*</span></label>
                                        <input
                                            type="number"
                                            id="arima-q"
                                            min="0"
                                            max="5"
                                            value={arimaParams.q}
                                            onChange={(e) => handleArimaParamChange('q', e.target.value)}
                                            required
                                            placeholder="Enter q value"
                                        />
                                    </div>
                                </div>

                                <h4>Seasonal Component</h4>
                                <div className="controls">
                                    <div className="control-group">
                                        <label htmlFor="arima-P">P (Seasonal AR) <span className="required-asterisk">*</span></label>
                                        <input
                                            type="number"
                                            id="arima-P"
                                            min="0"
                                            max="2"
                                            value={arimaParams.P}
                                            onChange={(e) => handleArimaParamChange('P', e.target.value)}
                                            required
                                            placeholder="Enter P value"
                                        />
                                    </div>
                                    <div className="control-group">
                                        <label htmlFor="arima-D">D (Seasonal Diff) <span className="required-asterisk">*</span></label>
                                        <input
                                            type="number"
                                            id="arima-D"
                                            min="0"
                                            max="1"
                                            value={arimaParams.D}
                                            onChange={(e) => handleArimaParamChange('D', e.target.value)}
                                            required
                                            placeholder="Enter D value"
                                        />
                                    </div>
                                    <div className="control-group">
                                        <label htmlFor="arima-Q">Q (Seasonal MA) <span className="required-asterisk">*</span></label>
                                        <input
                                            type="number"
                                            id="arima-Q"
                                            min="0"
                                            max="2"
                                            value={arimaParams.Q}
                                            onChange={(e) => handleArimaParamChange('Q', e.target.value)}
                                            required
                                            placeholder="Enter Q value"
                                        />
                                    </div>
                                    <div className="control-group">
                                        <label htmlFor="arima-s">s (Season Length) <span className="required-asterisk">*</span></label>
                                        <input
                                            type="number"
                                            id="arima-s"
                                            min="0"
                                            max="365"
                                            value={arimaParams.s}
                                            onChange={(e) => handleArimaParamChange('s', e.target.value)}
                                            required
                                            placeholder="Enter s value"
                                        />
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Re-analyze button */}
                        <button
                            className="analyze-btn"
                            onClick={analyzeData}
                            disabled={isLoading || !areArimaParamsValid()}
                            style={{ marginTop: '15px' }}
                        >
                            Re-analyze with Selected Options
                        </button>

                        {!areArimaParamsValid() && (
                            <p className="error-message" style={{ textAlign: 'center', marginTop: '10px' }}>
                                Please fill in all required ARIMA parameters
                            </p>
                        )}
                    </div>

                    <h4>Imputation:</h4>
                    <p className="step-instruction">Click "Impute" to apply the selected model and generate missing values</p>
                    <div className="button-group">
                        <button
                            className="analyze-btn inline-btn"
                            onClick={() => displayImputedData(datasetId, selectedYColumn)}
                            style={{ marginRight: '10px' }}
                            disabled={isLoading ||
                                (selectedModel === 'arima_imputation' && !areArimaParamsValid())}
                        >
                            Impute
                        </button>
                        {isImputed && (
                            <a
                                href={api.getDownloadUrl(datasetId)}
                                className="analyze-btn inline-btn"
                                download="imputed_data.csv"
                                target="_blank"
                                rel="noopener noreferrer"
                                title="Download imputed data as CSV"
                            >
                                <svg className="download-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M12 16L12 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                    <path d="M9 13L12 16L15 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                    <path d="M20 16V18C20 19.1046 19.1046 20 18 20H6C4.89543 20 4 19.1046 4 18L4 16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                </svg>
                                <span>CSV</span>
                            </a>
                        )}
                    </div>

                    {imputedData && (
                        <div>
                            <h4>Imputation Results:</h4>
                            <div className="chart-container">
                                <div
                                    ref={imputedChartRef}
                                    style={{ width: '100%', height: '100%' }}
                                ></div>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

window.TimeSeriesApp = App;