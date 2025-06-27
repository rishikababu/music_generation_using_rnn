document.getElementById('music-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    const inputType = document.querySelector('input[name="input-type"]:checked').value;
    const inputData = document.getElementById('input-data').value;
    const instrument = document.getElementById('instrument').value;
    const style = document.getElementById('style').value;
    const duration = document.getElementById('duration').value;
    const formData = new FormData();
    formData.append('type', inputType);
    formData.append('data', inputData);
    formData.append('instrument', instrument);
    formData.append('style', style);
    formData.append('duration', duration);
    const response = await fetch('/generate', {
        method: 'POST',
        body: formData
    });
    if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const audio = document.getElementById('audio');
        audio.src = url;
        audio.load();
        document.getElementById('download').href = url;
        // Show thanks message
        document.getElementById('thanks-message').style.display = 'block';
    } else {
        alert("Error generating music");
    }
});