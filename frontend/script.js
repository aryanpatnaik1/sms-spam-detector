const messageInput = document.getElementById("message");
const checkButton = document.getElementById("check-button");
const resultDisplay = document.getElementById("result");

checkButton.addEventListener("click", async function()
{
    const message = messageInput.value.trim();
    if(message === "")
    {
        resultDisplay.textContent = "Please enter a message.";
        return;
    }
    resultDisplay.textContent = "Checking message...";
    try{
        const payload = { message: message};
        const options = {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)};
        const response = await fetch("/predict", options);
        const data = await response.json();
        if(!response.ok)
        {
            resultDisplay.textContent = data.error;
            return;
        }
        checkButton.disabled = true;
        checkButton.textContent = "Checking...";
        const spampercentage = (data.spam_score*100).toFixed(1);
        resultDisplay.textContent = "Prediction: " + data.prediction + "| Spam Score: " + spampercentage + "%";
    } catch(error)
    {
        resultDisplay.textContent = "Could not check that message. Please try again.";
        console.error(error);   
    }
    finally
    {
        checkButton.disabled = false;
        checkButton.textContent = "Check Message";
    }
    const spampercentage = (data.spam_score*100).toFixed(1);
    resultDisplay.textContent = "Prediction: " + data.prediction + "| Spam Score: " + spampercentage + "%";
});