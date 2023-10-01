const formData = new FormData();
formData.append('image', file);

const jsonData = {
    image: formData.get('image')
};

const response = await fetch('/recognize', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json' // Set the content type to JSON
    },
    body: JSON.stringify(jsonData) // Convert the data to JSON format

    const data = await response.json();
    document.getElementById('result').innerText = data.predicted_name;
});
