const submit = document.querySelector('button');

submit.addEventListener('click', async () => {
    const inputColName = document.querySelector('input[name="input"]');
    const outputColName = document.querySelector('input[name="output"]');
    const testInput = document.querySelector('input[name="test"]');
    const file = document.querySelector('input[type="file"]');

    const outputHeading = document.querySelector('#outputHeading');
    const outputScore = document.querySelector('#score');
    const outputPrediction = document.querySelector('#output');

    const formData = new FormData();
    formData.append('file', file.files[0]);
    formData.append('input_column_name', inputColName.value);
    formData.append('output_column_name', outputColName.value);
    formData.append('test_input_value', testInput.value);

    try {
        const response = await fetch('http://127.0.0.1:8000/train_random_forest_regressor', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log(data);
        outputHeading.textContent = "Message: " + data.message;
        outputScore.textContent = "Score: " + data.score;
        outputPrediction.textContent = "GPA: " + data.gpa;
    } catch (error) {
        console.error('Error:', error);
    }
});
