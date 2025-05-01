const apiBaseUrl = typeof appConfig !== 'undefined' ? appConfig.apiBaseUrl : 'http://localhost:8000';

// Upload a dataset file to the server
const uploadDataset = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await axios.post(`${apiBaseUrl}/upload_dataset`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
    });

    return response.data;
};

// Preprocess the uploaded dataset
const preprocessDataset = async (datasetId) => {
    const response = await axios.post(`${apiBaseUrl}/preprocess`, {
        dataset_id: datasetId
    });

    return response.data;
};

// Get column information for a dataset
const getColumnInfo = async (datasetId) => {
    const response = await axios.get(`${apiBaseUrl}/dataset_columns/${datasetId}`);
    return response.data;
};

// Get dataset data for visualization
const getDatasetForChart = async (datasetId, xColumn, yColumn) => {
    let url = `${apiBaseUrl}/dataset/${datasetId}`;
    const params = new URLSearchParams();

    if (xColumn) {
        params.append('x_column', xColumn);
    }

    if (yColumn) {
        params.append('y_column', yColumn);
    }

    if (params.toString()) {
        url += `?${params.toString()}`;
    }

    const response = await axios.get(url);
    return response.data;
};

// Get dataset preview
const getDatasetPreview = async (datasetId) => {
    try {
        const response = await axios.get(`${apiBaseUrl}/dataset_preview/${datasetId}?rows=5`);

        if (!response) {
            response = await axios.get(`${apiBaseUrl}/dataset/${datasetId}`);
        }

        return response.data;
    } catch (error) {
        console.error("Error fetching dataset preview:", error);

        try {
            const chartData = await getDatasetForChart(datasetId);

            const previewData = {
                columns: ['Date', 'Value'],
                data: []
            };

            if (chartData && chartData.data && chartData.data.length > 0) {
                const sampleData = chartData.data.slice(0, 5);

                previewData.data = sampleData.map(point => ({
                    'Date': point.x,
                    'Value': point.y
                }));
            }

            return previewData;
        } catch (fallbackError) {
            console.error("Fallback failed:", fallbackError);
            throw error;
        }
    }
};

// Select the best imputation model
const selectModel = async (datasetId, overrideModel) => {
    const response = await axios.post(`${apiBaseUrl}/select_model`, {
        dataset_id: datasetId,
        override_model: overrideModel || undefined
    });

    return response.data;
};

// Perform imputation on the dataset
const imputeDataset = async (datasetId, selectedModel, selectedYColumn, modelParams = null) => {
    const requestData = {
        dataset_id: datasetId,
        selected_model: selectedModel,
        selected_y_column: selectedYColumn
    };

    // Include model parameters if provided
    if (modelParams) {
        if (selectedModel === 'arima_imputation') {
            requestData.model_params = {
                order: [modelParams.p, modelParams.d, modelParams.q],
                seasonal_order: [modelParams.P, modelParams.D, modelParams.Q, modelParams.s],
                y_column: selectedYColumn || modelParams.y_column,
                hasSeasonality: modelParams.hasSeasonality
            };
        } else if (selectedModel === 'linear_interpolation') {
            requestData.model_params = {
                method: modelParams.method
            };
        }
    }

    const response = await axios.post(`${apiBaseUrl}/impute`, requestData);
    return response.data;
};

// Get imputed dataset for visualization
const getImputedDataset = async (datasetId, selectedModel, selectedYColumn, modelParams = null) => {
    const requestData = {
        dataset_id: datasetId,
        selected_model: selectedModel,
        selected_y_column: selectedYColumn
    };

    // Include model parameters if provided
    if (modelParams) {
        if (selectedModel === 'arima_imputation') {
            requestData.model_params = {
                order: [modelParams.p, modelParams.d, modelParams.q],
                seasonal_order: [modelParams.P, modelParams.D, modelParams.Q, modelParams.s],
                y_column: selectedYColumn || modelParams.y_column,
                hasSeasonality: modelParams.hasSeasonality
            };
        } else if (selectedModel === 'linear_interpolation') {
            requestData.model_params = {
                method: modelParams.method
            };
        }
    }

    const response = await axios.post(`${apiBaseUrl}/imputed_dataset`, requestData);
    return response.data;
};

// Get the download URL for an imputed dataset
const getDownloadUrl = (datasetId) => {
    return `${apiBaseUrl}/download_imputed/${datasetId}`;
};

// Get ARIMA diagnostic plots
const getArimaDiagnostics = async (datasetId, column) => {
    let url = `${apiBaseUrl}/arima_diagnostics/${datasetId}`;

    if (column) {
        url += `?column=${column}`;
    }

    const response = await axios.get(url);
    return response.data;
};

const getGBHistogram = async (datasetId, column) => {
    let url = `${apiBaseUrl}/gb_histogram/${datasetId}`;

    if (column) {
        url += `?yColumn=${column}`;
    }

    const response = await axios.get(url);
    return response.data;
};

const TimeSeriesApi = {
    uploadDataset,
    preprocessDataset,
    getColumnInfo,
    getDatasetForChart,
    selectModel,
    imputeDataset,
    getImputedDataset,
    getDownloadUrl,
    getDatasetPreview,
    getArimaDiagnostics,
    getGBHistogram,
    apiBaseUrl
};