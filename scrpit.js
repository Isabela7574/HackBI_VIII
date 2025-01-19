document.getElementById('submit-button').addEventListener('click', async () => {
    const form = document.getElementById('upload-form');
    const formData = new FormData(form);

    try {
        console.log("Debug: Submitting form data..."); // Log start of submission
        const response = await fetch('http://127.0.0.1:8080/process-image', {
            method: 'POST',
            body: formData,
        });

        console.log("Debug: Response received"); // Log when a response is received
        if (!response.ok) {
            const errorText = await response.text();
            console.error("Debug: Server error response:", errorText);
            throw new Error(errorText || 'Unknown server error');
        }

        const result = await response.json();
        console.log("Debug: Prediction result:", result); // Log the result
        document.getElementById('result').innerText = `Predicted Letter: ${result.predicted_letter}`;
    } catch (error) {
        console.error('Debug: Error occurred:', error.message);
        document.getElementById('result').innerText = `Error: ${error.message}`;
    }
});
