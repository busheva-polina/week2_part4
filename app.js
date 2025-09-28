// Titanic Binary Classifier using TensorFlow.js
// Schema Definition - SWAP THESE VARIABLES FOR OTHER DATASETS
const TARGET_COLUMN = 'Survived'; // Binary target variable
const FEATURE_COLUMNS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']; // Feature columns
const ID_COLUMN = 'PassengerId'; // Identifier column (excluded from features)
const CATEGORICAL_COLUMNS = ['Sex', 'Pclass', 'Embarked']; // Categorical columns for one-hot encoding
const NUMERICAL_COLUMNS = ['Age', 'Fare', 'SibSp', 'Parch']; // Numerical columns for standardization

// Global variables
let rawTrainData = null;
let rawTestData = null;
let processedTrainData = null;
let processedTestData = null;
let model = null;
let trainingHistory = null;
let validationData = null;
let validationLabels = null;
let validationPredictions = null;
let currentMetrics = null;

// DOM elements
const loadDataBtn = document.getElementById('load-data-btn');
const preprocessBtn = document.getElementById('preprocess-btn');
const createModelBtn = document.getElementById('create-model-btn');
const trainBtn = document.getElementById('train-btn');
const evaluateBtn = document.getElementById('evaluate-btn');
const predictBtn = document.getElementById('predict-btn');
const exportModelBtn = document.getElementById('export-model-btn');
const thresholdSlider = document.getElementById('threshold-slider');
const thresholdValue = document.getElementById('threshold-value');

// Event listeners
loadDataBtn.addEventListener('click', loadAndInspectData);
preprocessBtn.addEventListener('click', preprocessData);
createModelBtn.addEventListener('click', createModel);
trainBtn.addEventListener('click', trainModel);
evaluateBtn.addEventListener('click', evaluateModel);
predictBtn.addEventListener('click', predictTestData);
exportModelBtn.addEventListener('click', exportModel);
thresholdSlider.addEventListener('input', updateThreshold);

// Load and inspect data from CSV files
async function loadAndInspectData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    
    if (!trainFile || !testFile) {
        alert('Please upload both train.csv and test.csv files');
        return;
    }
    
    try {
        // Load train data
        const trainText = await readFile(trainFile);
        rawTrainData = parseCSV(trainText);
        
        // Load test data
        const testText = await readFile(testFile);
        rawTestData = parseCSV(testText);
        
        // Display data preview
        displayDataPreview();
        
        // Calculate and display data statistics
        displayDataStats();
        
        // Enable preprocessing button
        preprocessBtn.disabled = false;
        
    } catch (error) {
        alert('Error loading data: ' + error.message);
        console.error(error);
    }
}

// Read file as text
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsText(file);
    });
}

// Robust CSV parser that handles commas in fields and quoted values
function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    if (lines.length === 0) return [];
    
    // Extract headers from first line
    const headers = parseCSVLine(lines[0]);
    
    // Parse data rows
    const data = [];
    for (let i = 1; i < lines.length; i++) {
        if (lines[i].trim() === '') continue; // Skip empty lines
        
        const values = parseCSVLine(lines[i]);
        const row = {};
        
        headers.forEach((header, index) => {
            // Use empty string for missing values, otherwise trim the value
            row[header.trim()] = index < values.length ? values[index].trim() : '';
        });
        
        data.push(row);
    }
    
    return data;
}

// Parse a single CSV line, handling quotes and commas within fields
function parseCSVLine(line) {
    const values = [];
    let currentValue = '';
    let inQuotes = false;
    let quoteChar = null;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        const nextChar = line[i + 1];
        
        if (!inQuotes) {
            if (char === '"' || char === "'") {
                // Start of quoted field
                inQuotes = true;
                quoteChar = char;
            } else if (char === ',') {
                // End of field
                values.push(currentValue);
                currentValue = '';
            } else {
                currentValue += char;
            }
        } else {
            // We are inside quotes
            if (char === quoteChar) {
                // Check if this is an escaped quote (two consecutive quotes)
                if (nextChar === quoteChar) {
                    currentValue += quoteChar;
                    i++; // Skip next quote
                } else {
                    // End of quoted field
                    inQuotes = false;
                }
            } else {
                currentValue += char;
            }
        }
    }
    
    // Add the last value
    values.push(currentValue);
    
    return values;
}

// Display data preview table
function displayDataPreview() {
    const previewDiv = document.getElementById('data-preview');
    previewDiv.innerHTML = '<h3>Data Preview</h3>';
    
    if (!rawTrainData || rawTrainData.length === 0) {
        previewDiv.innerHTML += '<p>No data loaded</p>';
        return;
    }
    
    // Train data preview
    const trainTable = document.createElement('table');
    trainTable.innerHTML = `<caption>Train Data (First 5 rows)</caption>`;
    
    // Create header - only show relevant columns for better display
    const displayColumns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'];
    let headerRow = '<tr>' + displayColumns.map(h => `<th>${h}</th>`).join('') + '</tr>';
    trainTable.innerHTML += headerRow;
    
    // Create data rows
    rawTrainData.slice(0, 5).forEach(row => {
        const dataRow = '<tr>' + displayColumns.map(h => `<td>${row[h] || ''}</td>`).join('') + '</tr>';
        trainTable.innerHTML += dataRow;
    });
    
    previewDiv.appendChild(trainTable);
    
    // Test data preview
    if (rawTestData && rawTestData.length > 0) {
        const testTable = document.createElement('table');
        testTable.innerHTML = `<caption>Test Data (First 5 rows)</caption>`;
        testTable.innerHTML += headerRow;
        
        rawTestData.slice(0, 5).forEach(row => {
            const dataRow = '<tr>' + displayColumns.map(h => `<td>${row[h] || ''}</td>`).join('') + '</tr>';
            testTable.innerHTML += dataRow;
        });
        
        previewDiv.appendChild(testTable);
    }
}

// Display data statistics and visualizations
function displayDataStats() {
    const previewDiv = document.getElementById('data-preview');
    
    // Calculate shapes
    const trainShape = `Train: ${rawTrainData.length} rows, ${Object.keys(rawTrainData[0]).length} columns`;
    const testShape = rawTestData ? `Test: ${rawTestData.length} rows, ${Object.keys(rawTestData[0]).length} columns` : 'Test: No data';
    
    // Calculate missing values percentage
    const trainMissing = calculateMissingPercentage(rawTrainData);
    const testMissing = rawTestData ? calculateMissingPercentage(rawTestData) : {};
    
    // Display shapes and missing values
    const statsDiv = document.createElement('div');
    statsDiv.innerHTML = `
        <h3>Data Statistics</h3>
        <p><strong>Shape:</strong> ${trainShape}, ${testShape}</p>
        <p><strong>Missing Values (%):</strong></p>
        <ul>
            ${Object.entries(trainMissing).map(([col, percent]) => 
                `<li>Train ${col}: ${percent.toFixed(2)}%</li>`
            ).join('')}
            ${Object.entries(testMissing).map(([col, percent]) => 
                `<li>Test ${col}: ${percent.toFixed(2)}%</li>`
            ).join('')}
        </ul>
    `;
    previewDiv.appendChild(statsDiv);
    
    // Create survival visualizations
    createSurvivalVisualizations();
}

// Calculate missing values percentage
function calculateMissingPercentage(data) {
    const missing = {};
    const totalRows = data.length;
    
    Object.keys(data[0]).forEach(column => {
        const missingCount = data.filter(row => !row[column] || row[column] === '').length;
        missing[column] = (missingCount / totalRows) * 100;
    });
    
    return missing;
}

// Create survival rate visualizations using tfjs-vis
function createSurvivalVisualizations() {
    // Survival by Sex
    const survivalBySex = {};
    rawTrainData.forEach(row => {
        const sex = row.Sex;
        const survived = parseInt(row.Survived);
        
        if (!survivalBySex[sex]) {
            survivalBySex[sex] = { total: 0, survived: 0 };
        }
        
        survivalBySex[sex].total++;
        if (survived === 1) survivalBySex[sex].survived++;
    });
    
    const sexData = Object.entries(survivalBySex).map(([sex, stats]) => ({
        x: sex,
        y: (stats.survived / stats.total) * 100
    }));
    
    // Survival by Pclass
    const survivalByPclass = {};
    rawTrainData.forEach(row => {
        const pclass = row.Pclass;
        const survived = parseInt(row.Survived);
        
        if (!survivalByPclass[pclass]) {
            survivalByPclass[pclass] = { total: 0, survived: 0 };
        }
        
        survivalByPclass[pclass].total++;
        if (survived === 1) survivalByPclass[pclass].survived++;
    });
    
    const pclassData = Object.entries(survivalByPclass).map(([pclass, stats]) => ({
        x: `Class ${pclass}`,
        y: (stats.survived / stats.total) * 100
    }));
    
    // Create visualizations
    tfvis.render.barchart(
        { name: 'Survival Rates by Sex', tab: 'Data Analysis' },
        [{ values: sexData }],
        { xLabel: 'Sex', yLabel: 'Survival Rate (%)' }
    );
    
    tfvis.render.barchart(
        { name: 'Survival by Passenger Class', tab: 'Data Analysis' }, 
        [{ values: pclassData }],
        { xLabel: 'Passenger Class', yLabel: 'Survival Rate (%)' }
    );
}

// Preprocess the data
function preprocessData() {
    try {
        // Process train data
        processedTrainData = {
            features: [],
            labels: [],
            passengerIds: []
        };
        
        // Process test data
        processedTestData = {
            features: [],
            passengerIds: []
        };
        
        // Calculate imputation values from train data
        const imputationValues = calculateImputationValues(rawTrainData);
        
        // Calculate standardization parameters from train data
        const standardizationParams = calculateStandardizationParams(rawTrainData);
        
        // Process train data
        rawTrainData.forEach(row => {
            const features = extractFeatures(row, imputationValues, standardizationParams);
            processedTrainData.features.push(features);
            processedTrainData.labels.push(parseInt(row[TARGET_COLUMN]));
            processedTrainData.passengerIds.push(row[ID_COLUMN]);
        });
        
        // Process test data
        rawTestData.forEach(row => {
            const features = extractFeatures(row, imputationValues, standardizationParams);
            processedTestData.features.push(features);
            processedTestData.passengerIds.push(row[ID_COLUMN]);
        });
        
        // Convert to tensors
        processedTrainData.features = tf.tensor2d(processedTrainData.features);
        processedTrainData.labels = tf.tensor1d(processedTrainData.labels);
        
        processedTestData.features = tf.tensor2d(processedTestData.features);
        
        // Display preprocessing results
        displayPreprocessingResults();
        
        // Enable model creation button
        createModelBtn.disabled = false;
        
    } catch (error) {
        alert('Error preprocessing data: ' + error.message);
        console.error(error);
    }
}

// Calculate imputation values (median for Age, mode for Embarked)
function calculateImputationValues(data) {
    const imputation = {};
    
    // Age: median
    const ages = data.map(row => parseFloat(row.Age)).filter(age => !isNaN(age));
    imputation.Age = ages.length > 0 ? 
        ages.sort((a, b) => a - b)[Math.floor(ages.length / 2)] : 0;
    
    // Embarked: mode
    const embarkedCounts = {};
    data.forEach(row => {
        if (row.Embarked && row.Embarked !== '') {
            embarkedCounts[row.Embarked] = (embarkedCounts[row.Embarked] || 0) + 1;
        }
    });
    imputation.Embarked = Object.keys(embarkedCounts).length > 0 ?
        Object.keys(embarkedCounts).reduce((a, b) => embarkedCounts[a] > embarkedCounts[b] ? a : b) : 'S';
    
    // Fare: median (for test data)
    const fares = data.map(row => parseFloat(row.Fare)).filter(fare => !isNaN(fare));
    imputation.Fare = fares.length > 0 ? 
        fares.sort((a, b) => a - b)[Math.floor(fares.length / 2)] : 0;
    
    return imputation;
}

// Calculate standardization parameters (mean and std)
function calculateStandardizationParams(data) {
    const params = {};
    
    NUMERICAL_COLUMNS.forEach(col => {
        const values = data.map(row => {
            const val = parseFloat(row[col]);
            return isNaN(val) ? null : val;
        }).filter(val => val !== null);
        
        if (values.length > 0) {
            const mean = values.reduce((a, b) => a + b, 0) / values.length;
            const std = Math.sqrt(values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length);
            params[col] = { mean, std: std || 1 }; // Avoid division by zero
        } else {
            params[col] = { mean: 0, std: 1 };
        }
    });
    
    return params;
}

// Extract and preprocess features from a row
function extractFeatures(row, imputationValues, standardizationParams) {
    const features = [];
    
    // Handle numerical features
    NUMERICAL_COLUMNS.forEach(col => {
        let value = parseFloat(row[col]);
        if (isNaN(value)) {
            // Impute missing values
            value = imputationValues[col] || 0;
        }
        
        // Standardize
        if (standardizationParams[col]) {
            value = (value - standardizationParams[col].mean) / standardizationParams[col].std;
        }
        
        features.push(value);
    });
    
    // Handle categorical features (one-hot encoding)
    CATEGORICAL_COLUMNS.forEach(col => {
        let value = row[col];
        
        // Impute missing values for Embarked
        if (col === 'Embarked' && (!value || value === '')) {
            value = imputationValues.Embarked;
        }
        
        // One-hot encoding
        if (col === 'Sex') {
            features.push(value === 'female' ? 1 : 0); // Female: 1, Male: 0
        } else if (col === 'Pclass') {
            // One-hot for Pclass (3 classes)
            const pclass = parseInt(value);
            features.push(pclass === 1 ? 1 : 0);
            features.push(pclass === 2 ? 1 : 0);
            features.push(pclass === 3 ? 1 : 0);
        } else if (col === 'Embarked') {
            // One-hot for Embarked (3 classes: C, Q, S)
            features.push(value === 'C' ? 1 : 0);
            features.push(value === 'Q' ? 1 : 0);
            features.push(value === 'S' ? 1 : 0);
        }
    });
    
    // Optional feature engineering
    // FamilySize = SibSp + Parch + 1
    const sibsp = parseInt(row.SibSp) || 0;
    const parch = parseInt(row.Parch) || 0;
    const familySize = sibsp + parch + 1;
    features.push(familySize);
    
    // IsAlone = (FamilySize == 1)
    features.push(familySize === 1 ? 1 : 0);
    
    return features;
}

// Display preprocessing results
function displayPreprocessingResults() {
    const outputDiv = document.getElementById('preprocessing-output');
    
    outputDiv.innerHTML = `
        <h3>Preprocessing Complete</h3>
        <p><strong>Train Features Shape:</strong> ${processedTrainData.features.shape}</p>
        <p><strong>Train Labels Shape:</strong> ${processedTrainData.labels.shape}</p>
        <p><strong>Test Features Shape:</strong> ${processedTestData.features.shape}</p>
        <p><strong>Number of Features:</strong> ${processedTrainData.features.shape[1]}</p>
        <p><strong>Feature Names:</strong> ${NUMERICAL_COLUMNS.join(', ')}, Sex, Pclass_1, Pclass_2, Pclass_3, Embarked_C, Embarked_Q, Embarked_S, FamilySize, IsAlone</p>
    `;
}

// Create the neural network model
function createModel() {
    try {
        const numFeatures = processedTrainData.features.shape[1];
        
        model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [numFeatures],
                    units: 16,
                    activation: 'relu'
                }),
                tf.layers.dense({
                    units: 1,
                    activation: 'sigmoid'
                })
            ]
        });
        
        model.compile({
            optimizer: 'adam',
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        // Display model summary
        displayModelSummary();
        
        // Enable training button
        trainBtn.disabled = false;
        
    } catch (error) {
        alert('Error creating model: ' + error.message);
        console.error(error);
    }
}

// Display model summary
function displayModelSummary() {
    const summaryDiv = document.getElementById('model-summary');
    
    // Clear previous content
    summaryDiv.innerHTML = '<h3>Model Summary</h3>';
    
    // Create a simple summary since tfjs doesn't have a built-in summary function for the browser
    const modelSummary = [];
    model.layers.forEach((layer, i) => {
        modelSummary.push({
            'Layer': layer.name,
            'Type': layer.getClassName(),
            'Output Shape': layer.outputShape.join(' x '),
            'Parameters': layer.countParams()
        });
    });
    
    const totalParams = model.trainableWeights
        .map(w => w.shape.reduce((a, b) => a * b))
        .reduce((a, b) => a + b, 0);
    
    const table = document.createElement('table');
    table.innerHTML = `
        <tr>
            <th>Layer</th>
            <th>Type</th>
            <th>Output Shape</th>
            <th>Parameters</th>
        </tr>
        ${modelSummary.map(layer => `
            <tr>
                <td>${layer.Layer}</td>
                <td>${layer.Type}</td>
                <td>${layer['Output Shape']}</td>
                <td>${layer.Parameters}</td>
            </tr>
        `).join('')}
        <tr>
            <td colspan="3"><strong>Total Parameters</strong></td>
            <td><strong>${totalParams}</strong></td>
        </tr>
    `;
    
    summaryDiv.appendChild(table);
}

// Train the model
async function trainModel() {
    try {
        const statusDiv = document.getElementById('training-status');
        statusDiv.innerHTML = '<p>Training started... This may take a while.</p>';
        
        // Create validation split (80/20 stratified)
        const { trainFeatures, trainLabels, valFeatures, valLabels } = createValidationSplit();
        validationData = valFeatures;
        validationLabels = valLabels;
        
        // Training parameters
        const epochs = 50;
        const batchSize = 32;
        
        // Train the model with tfjs-vis callbacks
        trainingHistory = await model.fit(trainFeatures, trainLabels, {
            epochs: epochs,
            batchSize: batchSize,
            validationData: [valFeatures, valLabels],
            callbacks: tfvis.show.fitCallbacks(
                { name: 'Training Performance' },
                ['loss', 'val_loss', 'acc', 'val_acc'],
                { callbacks: ['onEpochEnd'] }
            )
        });
        
        statusDiv.innerHTML += '<p>Training completed!</p>';
        
        // Enable evaluation and prediction buttons
        evaluateBtn.disabled = false;
        predictBtn.disabled = false;
        exportModelBtn.disabled = false;
        thresholdSlider.disabled = false;
        
    } catch (error) {
        alert('Error training model: ' + error.message);
        console.error(error);
    }
}

// Create validation split (80/20 stratified)
function createValidationSplit() {
    const features = processedTrainData.features;
    const labels = processedTrainData.labels;
    
    // Simple random split (for simplicity, in production you'd want stratified split)
    const splitIndex = Math.floor(features.shape[0] * 0.8);
    
    const trainFeatures = features.slice(0, splitIndex);
    const trainLabels = labels.slice(0, splitIndex);
    const valFeatures = features.slice(splitIndex);
    const valLabels = labels.slice(splitIndex);
    
    return { trainFeatures, trainLabels, valFeatures, valLabels };
}

// Evaluate the model and compute metrics
async function evaluateModel() {
    try {
        const metricsDiv = document.getElementById('metrics-output');
        metricsDiv.innerHTML = '<p>Evaluating model...</p>';
        
        // Get predictions on validation set
        validationPredictions = model.predict(validationData);
        const probs = await validationPredictions.data();
        const labels = await validationLabels.data();
        
        // Compute ROC curve and AUC
        const { rocCurve, auc } = computeROC(labels, probs);
        
        // Plot ROC curve
        plotROCCurve(rocCurve, auc);
        
        // Update metrics with default threshold
        updateThreshold();
        
        metricsDiv.innerHTML = '<p>Evaluation complete! Use the slider to adjust classification threshold.</p>';
        
    } catch (error) {
        alert('Error evaluating model: ' + error.message);
        console.error(error);
    }
}

// Compute ROC curve and AUC
function computeROC(labels, probabilities) {
    // Create thresholds from 0 to 1
    const thresholds = Array.from({ length: 101 }, (_, i) => i / 100);
    
    const rocPoints = thresholds.map(threshold => {
        let tp = 0, fp = 0, tn = 0, fn = 0;
        
        for (let i = 0; i < labels.length; i++) {
            const prediction = probabilities[i] >= threshold ? 1 : 0;
            const actual = labels[i];
            
            if (prediction === 1 && actual === 1) tp++;
            else if (prediction === 1 && actual === 0) fp++;
            else if (prediction === 0 && actual === 0) tn++;
            else if (prediction === 0 && actual === 1) fn++;
        }
        
        const tpr = tp / (tp + fn) || 0; // True Positive Rate (Sensitivity)
        const fpr = fp / (fp + tn) || 0; // False Positive Rate (1 - Specificity)
        
        return { threshold, tpr, fpr, tp, fp, tn, fn };
    });
    
    // Calculate AUC using trapezoidal rule
    let auc = 0;
    for (let i = 1; i < rocPoints.length; i++) {
        const prev = rocPoints[i - 1];
        const curr = rocPoints[i];
        auc += (curr.fpr - prev.fpr) * (curr.tpr + prev.tpr) / 2;
    }
    
    return { rocCurve: rocPoints, auc: Math.abs(auc) };
}

// Plot ROC curve using tfjs-vis
function plotROCCurve(rocCurve, auc) {
    const rocData = rocCurve.map(point => ({
        x: point.fpr,
        y: point.tpr
    }));
    
    tfvis.render.scatterplot(
        { name: `ROC Curve (AUC = ${auc.toFixed(3)})`, tab: 'Evaluation' },
        { values: [rocData] },
        {
            xLabel: 'False Positive Rate',
            yLabel: 'True Positive Rate',
            series: ['ROC Curve'],
            height: 400
        }
    );
}

// Update metrics based on threshold slider
function updateThreshold() {
    if (!validationPredictions || !validationLabels) return;
    
    const threshold = parseFloat(thresholdSlider.value);
    thresholdValue.textContent = threshold.toFixed(2);
    
    // Compute metrics with current threshold
    computeMetrics(threshold);
}

// Compute classification metrics
async function computeMetrics(threshold) {
    const probs = await validationPredictions.data();
    const labels = await validationLabels.data();
    
    let tp = 0, fp = 0, tn = 0, fn = 0;
    
    for (let i = 0; i < labels.length; i++) {
        const prediction = probs[i] >= threshold ? 1 : 0;
        const actual = labels[i];
        
        if (prediction === 1 && actual === 1) tp++;
        else if (prediction === 1 && actual === 0) fp++;
        else if (prediction === 0 && actual === 0) tn++;
        else if (prediction === 0 && actual === 1) fn++;
    }
    
    const accuracy = (tp + tn) / (tp + fp + tn + fn);
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    
    currentMetrics = { tp, fp, tn, fn, accuracy, precision, recall, f1, threshold };
    displayMetrics(currentMetrics);
    plotConfusionMatrix({ tp, fp, tn, fn });
}

// Display metrics in a proper table format
function displayMetrics(metrics) {
    const metricsDiv = document.getElementById('metrics-output');
    
    metricsDiv.innerHTML = `
        <div class="metrics-container">
            <div class="metric-card">
                <h3>Confusion Matrix</h3>
                <table style="width: 100%; margin: 10px 0;">
                    <tr>
                        <th></th>
                        <th>Predicted Negative</th>
                        <th>Predicted Positive</th>
                    </tr>
                    <tr>
                        <th>Actual Negative</th>
                        <td>${metrics.tn} (TN)</td>
                        <td>${metrics.fp} (FP)</td>
                    </tr>
                    <tr>
                        <th>Actual Positive</th>
                        <td>${metrics.fn} (FN)</td>
                        <td>${metrics.tp} (TP)</td>
                    </tr>
                </table>
                <p><strong>Threshold:</strong> ${metrics.threshold.toFixed(2)}</p>
            </div>
            <div class="metric-card">
                <h3>Performance Metrics</h3>
                <table style="width: 100%; margin: 10px 0;">
                    <tr>
                        <td><strong>Accuracy:</strong></td>
                        <td>${metrics.accuracy.toFixed(4)}</td>
                    </tr>
                    <tr>
                        <td><strong>Precision:</strong></td>
                        <td>${metrics.precision.toFixed(4)}</td>
                    </tr>
                    <tr>
                        <td><strong>Recall:</strong></td>
                        <td>${metrics.recall.toFixed(4)}</td>
                    </tr>
                    <tr>
                        <td><strong>F1-Score:</strong></td>
                        <td>${metrics.f1.toFixed(4)}</td>
                    </tr>
                </table>
                <p><strong>Total Samples:</strong> ${metrics.tp + metrics.fp + metrics.tn + metrics.fn}</p>
            </div>
        </div>
    `;
}

// Plot confusion matrix using tfjs-vis
function plotConfusionMatrix({ tp, fp, tn, fn }) {
    const confusionMatrix = [
        [tn, fp],
        [fn, tp]
    ];
    
    const classNames = ['Not Survived', 'Survived'];
    
    tfvis.render.confusionMatrix(
        { name: 'Confusion Matrix', tab: 'Evaluation' },
        { values: confusionMatrix, labels: classNames },
        { height: 400 }
    );
}

// Predict on test data
async function predictTestData() {
    try {
        const outputDiv = document.getElementById('prediction-output');
        outputDiv.innerHTML = '<p>Generating predictions...</p>';
        
        // Get predictions
        const testPredictions = model.predict(processedTestData.features);
        const probabilities = await testPredictions.data();
        
        // Apply threshold to get binary predictions
        const threshold = parseFloat(thresholdSlider.value);
        const binaryPredictions = probabilities.map(p => p >= threshold ? 1 : 0);
        
        // Create submission file
        createSubmissionFile(binaryPredictions, probabilities);
        
        outputDiv.innerHTML = `
            <p>Predictions generated!</p>
            <p>Survived: ${binaryPredictions.filter(p => p === 1).length} passengers</p>
            <p>Not Survived: ${binaryPredictions.filter(p => p === 0).length} passengers</p>
            <p>Files downloaded: submission.csv, probabilities.csv</p>
        `;
        
    } catch (error) {
        alert('Error generating predictions: ' + error.message);
        console.error(error);
    }
}

// Create submission file for download
function createSubmissionFile(predictions, probabilities) {
    // Create submission CSV (PassengerId, Survived)
    let submissionCSV = 'PassengerId,Survived\n';
    let probabilitiesCSV = 'PassengerId,Probability\n';
    
    processedTestData.passengerIds.forEach((passengerId, index) => {
        submissionCSV += `${passengerId},${predictions[index]}\n`;
        probabilitiesCSV += `${passengerId},${probabilities[index].toFixed(6)}\n`;
    });
    
    // Download files
    downloadFile(submissionCSV, 'submission.csv', 'text/csv');
    downloadFile(probabilitiesCSV, 'probabilities.csv', 'text/csv');
}

// Download file utility
function downloadFile(content, filename, contentType) {
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Export model
async function exportModel() {
    try {
        await model.save('downloads://titanic-tfjs-model');
        alert('Model exported successfully!');
    } catch (error) {
        alert('Error exporting model: ' + error.message);
        console.error(error);
    }
}
