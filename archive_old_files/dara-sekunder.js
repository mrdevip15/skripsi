// Function to format date as DD-MM-YYYY
function formatDate(date) {
    const day = String(date.getDate()).padStart(2, '0');
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const year = date.getFullYear();
    return `${day}-${month}-${year}`;
}

// Function to get month-year string for logging
function getMonthYearString(date) {
    const month = date.toLocaleString('default', { month: 'long' });
    const year = date.getFullYear();
    return `${month} ${year}`;
}

// Function to get last day of month
function getLastDayOfMonth(date) {
    return new Date(date.getFullYear(), date.getMonth() + 1, 0);
}

// Function to download data for a specific month
async function downloadMonthData(startDate) {
    const endDate = getLastDayOfMonth(startDate);
    const monthYear = getMonthYearString(startDate);
    
    console.log(`Starting download for ${monthYear}...`);
    console.log(`Date range: ${formatDate(startDate)} to ${formatDate(endDate)}`);
    
    document.getElementsByName("from")[0].value = formatDate(startDate);
    document.getElementsByName("to")[0].value = formatDate(endDate);
    
    $("#submitButton").click();
    console.log('Submit button clicked, waiting 2 seconds...');
    
    // Wait for 2 seconds
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    $("#excelButton").click();
    console.log(`Excel download initiated for ${monthYear}`);
    console.log('Note: Please rename the downloaded file to: ' + 
        `${String(startDate.getMonth() + 1).padStart(2, '0')}-${startDate.getFullYear()}.xlsx`);
    
    // Wait for 2 seconds after excel download
    await new Promise(resolve => setTimeout(resolve, 2000));
    console.log('Download completed for ' + monthYear);
    console.log('----------------------------------------');
}

// Function to create filename summary table
function createFilenameTable(startDate, endDate) {
    console.log('\nFilename Summary Table:');
    console.log('----------------------------------------');
    console.log('Original Filename\t\tRename To');
    console.log('----------------------------------------');
    
    let currentDate = new Date(startDate);
    while (currentDate <= endDate) {
        const month = String(currentDate.getMonth() + 1).padStart(2, '0');
        const year = currentDate.getFullYear();
        const monthName = currentDate.toLocaleString('default', { month: 'long' });
        
        // Assuming the original filename is something like "data_export.xlsx"
        console.log(`data_export.xlsx\t\t\t${month}-${year}.xlsx (${monthName} ${year})`);
        
        currentDate.setMonth(currentDate.getMonth() + 1);
    }
    console.log('----------------------------------------');
}

// Main function to download all data
async function downloadAllData() {
    const startDate = new Date(2010, 0, 1); // January 1, 2010
    const endDate = new Date(2025, 4, 24); // May 24, 2025
    
    let currentDate = new Date(startDate);
    let totalMonths = 0;
    
    console.log('Starting bulk download process...');
    console.log('Total date range: ' + formatDate(startDate) + ' to ' + formatDate(endDate));
    console.log('----------------------------------------');
    
    while (currentDate <= endDate) {
        await downloadMonthData(currentDate);
        totalMonths++;
        
        // Move to next month
        currentDate.setMonth(currentDate.getMonth() + 1);
    }
    
    console.log('----------------------------------------');
    console.log(`Download process completed!`);
    console.log(`Total months downloaded: ${totalMonths}`);
    console.log('Please rename your downloaded files to the format: MM-YYYY.xlsx');
    
    // Create and display the filename summary table
    createFilenameTable(startDate, endDate);
}

// Start the download process
downloadAllData();
