async function classifyNews() {
    let newsText = document.getElementById("newsInput").value;
    let resultDiv = document.getElementById("result");

    if (!newsText.trim()) {
        resultDiv.innerHTML = "<p style='color:yellow;'>Please enter news text!</p>";
        return;
    }

    try {
        let response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: newsText }),
        });

        if (!response.ok) {
            throw new Error("Server error, please try again.");
        }

        let data = await response.json();
        resultDiv.innerHTML = `<p><strong>Bias Type:</strong> ${data.classification.bias}</p>
                               `;
    } catch (error) {
        resultDiv.innerHTML = `<p style="color:yellow;">Error: ${error.message}</p>`;
    }
}
