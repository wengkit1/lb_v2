function setupDynamicAverages() {

    function recalculateAverages() {
        const allTables = document.querySelectorAll('table');

        allTables.forEach(table => {
            const tableRows = table.querySelectorAll('tbody tr');
            if (tableRows.length === 0) return;

            // Get all column headers
            const headerCells = table.querySelectorAll('thead th');
            const headers = Array.from(headerCells).map(th => th.textContent.trim());

            // Find which columns we need to update
            const seaAvgColumnIndex = headers.findIndex(h => h === 'SEA Avg');
            const totalAvgColumnIndex = headers.findIndex(h => h === 'Aggregate');

            if (seaAvgColumnIndex === -1 && totalAvgColumnIndex === -1) return;

            const seaLanguageNames = ['Indonesian', 'Vietnamese', 'Thai', 'Tamil', 'Tagalog'];

            // Process each row in the table
            tableRows.forEach(row => {
                const rowCells = row.querySelectorAll('td');
                if (rowCells.length === 0) return;

                let totalSum = 0, totalCount = 0;
                let seaSum = 0, seaCount = 0;

                // Go through each header and sum values
                headers.forEach((headerText, index) => {
                    if (headerText === 'model' || headerText === 'SEA Avg' || headerText === 'Aggregate') {
                        return;
                    }

                    const cellValue = parseFloat(rowCells[index]?.textContent);
                    if (!isNaN(cellValue)) {
                        totalSum += cellValue;
                        totalCount++;

                        if (seaLanguageNames.includes(headerText)) {
                            seaSum += cellValue;
                            seaCount++;
                        }
                    }
                });

                // Update SEA Average column
                if (seaAvgColumnIndex !== -1 && rowCells[seaAvgColumnIndex]) {
                    rowCells[seaAvgColumnIndex].textContent =
                        seaCount > 0 ? (seaSum / seaCount).toFixed(2) : '0.00';
                }

                // Update Total Average column
                if (totalAvgColumnIndex !== -1 && rowCells[totalAvgColumnIndex]) {
                    rowCells[totalAvgColumnIndex].textContent =
                        totalCount > 0 ? (totalSum / totalCount).toFixed(2) : '0.00';
                }
            });
        });
    }

    function setupChangeObserver() {
        const observer = new MutationObserver(() => {
            setTimeout(recalculateAverages, 50);
        });

        const gradioContainer = document.querySelector('.gradio-container');
        if (gradioContainer) {
            observer.observe(gradioContainer, {
                childList: true,
                subtree: true,
                attributes: true,
                attributeFilter: ['style', 'hidden']
            });
        }
    }

    setTimeout(() => {
        setupChangeObserver();
        recalculateAverages();
    }, 1000);
}

setupDynamicAverages();