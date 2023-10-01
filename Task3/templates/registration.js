document.getElementById('registration-form').addEventListener('submit', async (event) => {
    event.preventDefault();

    const formData = new FormData();
    formData.append('name', document.getElementById('name').value);
    formData.append('email', document.getElementById('email').value);
    
    // Get the selected image file
    const imageFile = document.getElementById('image').files[0];
    
    // Check if an image file was selected
    if (imageFile) {
        // Read the image file as a data URL and convert it to a Base64 string
        const reader = new FileReader();
        reader.onloadend = async function() {
            const base64Image = reader.result.split(',')[1]; // Extract the Base64 data
            // Inside your reader.onloadend function
            //console.log('Base64 Image:', base64Image);
            
            // Add the Base64 image data to the form data
            formData.append('image', base64Image);
            // Before sending the request
            console.log('Form Data:', formData);

            try {
                const response = await fetch('/register', {
                    method: 'POST',
                    body: formData
                });

            const data = await response.json();
            alert(data.message);  // Display server response (for debugging)

            // After registering, retrain the model with the new data
            const retrainData = {
                name: document.getElementById('name').value,
                email: document.getElementById('email').value,
                image: base64Image // Use base64Image directly, not formData.get('image')
            };

            const retrainResponse = await fetch('/update_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(retrainData)
            });

            const retrainResponseData = await retrainResponse.json();
            alert(retrainResponseData.message);  // Display server response (for debugging)

            // Redirect to the recognition page after retraining
            if (retrainResponseData.message === 'Model updated successfully!') {
                window.location.href = '/recognize';
            }
        } catch (error) {
            console.error('Error:', error);
        }
    };

    // Read the image file as a data URL
    reader.readAsDataURL(imageFile);
    } else {
        alert('Please select an image file.');
    }
});
