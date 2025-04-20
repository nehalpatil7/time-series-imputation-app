// API Configuration
const config = {
    apiBaseUrl: 'http://localhost:8000',
};

if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
    // When deployed to cloud, use relative URL
    config.apiBaseUrl = '';
}

window.appConfig = config;