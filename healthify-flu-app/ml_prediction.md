# Seasonal Flu Hospitalization Prediction: ML Model Results

<a href="index.html" class="healthify-banner">Healthify</a>

<style>
/* Full-width Healthify Banner */
.healthify-banner {
    display: block;
    width: 100%;
    padding: 20px;
    font-size: 28px;
    font-weight: bold;
    color: white;
    background-color: #0047ab;
    text-decoration: none;
    text-align: center;
    border-radius: 0;
    transition: background 0.3s, transform 0.2s;
    position: fixed;
    top: 0;
    left: 0;
    z-index: 1000;
}
.healthify-banner:hover {
    background-color: #003580;
    transform: scale(1.03);
}
</style>

---

## **Methodology & Key Results Summary**

We developed a **logistic regression model** to classify **high vs. low seasonal flu hospitalization incidence** based on **age group** and **race category**. The dataset includes multiple flu seasons, and **SMOTE** was applied to address class imbalance before training (Rahim 2019, Huyen 2022).

- **Age-Based Model:** **74.3% accuracy**  
  - Performed well but had some misclassification of high-incidence cases.
- **Race-Based Model:** **76.9% accuracy**  
  - Slightly better performance, with higher recall for high-incidence cases.
- **Confusion Matrix Insights:**  
  - Most misclassifications were **false positives** (overestimating high incidence).
  - **Race appears to be a stronger predictor** than age alone.

---

## **ML Model Inference**
Age: <select id="age">
  <option value="">Age</option>
  <option value="0">0-17</option>
  <option value="1">18-49</option>
  <option value="2">50-64</option>
  <option value="3">65-74</option>
  <option value="4">75-84</option>
</select> 
<span id="age_result"></span>

Race: <select id="race">
  <option value="">Race</option>
  <option value="0">American Indian/Alaska Native</option>
  <option value="1">Asian/Pacific Islander</option>
  <option value="2">Black</option>
  <option value="3">Hispanic/Latino</option>
  <option value="4">White</option>
</select> 
<span id="race_result"></span>

<script>
    
    document.addEventListener('DOMContentLoaded', function() {
        document.getElementById('age').addEventListener('change', function() {
            const descriptionText = document.getElementById('age_result');
            const selectedAge = this.value;
    
            if (selectedAge) {
                descriptionText.textContent = 1/(1 + Math.exp(-0.24149078*selectedAge + 0.28622803)) >= 0.5 ? "High Incidence" : "Low Incidence" ;
                descriptionText.style.display = 'block';
            } else {
                descriptionText.textContent = "";
                descriptionText.style.display = 'none';
            }
        });
    
        document.getElementById('race').addEventListener('change', function() {
            const descriptionText = document.getElementById('race_result');
            const selectedRace = this.value;
    
            if (selectedRace) {
                descriptionText.textContent = 1/(1 + Math.exp(-0.0889156*selectedRace + 0.17704719)) >= 0.5 ? "High Incidence" : "Low Incidence";
                descriptionText.style.display = 'block';
            } else {
                descriptionText.textContent = "";
                descriptionText.style.display = 'none';
            }
        });
    });
</script>
---

## **Model Performance Metrics Table**
This table summarizes the accuracy, precision, recall, and F1-score for each model.

<iframe title="Flu Hospitalization ML Model Performance" 
        aria-label="Table" 
        id="datawrapper-chart-uEwrh" 
        src="https://datawrapper.dwcdn.net/Jr9zx/2/" 
        scrolling="no" frameborder="0" 
        style="width: 0; min-width: 100% !important; border: none;" 
        height="188" data-external="1">
</iframe>

<script type="text/javascript">
!function() {
    "use strict";
    window.addEventListener("message", (function(a) {
        if (void 0 !== a.data["datawrapper-height"]) {
            var e = document.querySelectorAll("iframe");
            for (var t in a.data["datawrapper-height"])
                for (var r = 0; r < e.length; r++)
                    if (e[r].contentWindow === a.source) {
                        var i = a.data["datawrapper-height"][t] + "px";
                        e[r].style.height = i;
                    }
        }
    }));
}();
</script>

---

## **Confusion Matrix: Model Predictions vs. Actual Data**
The confusion matrix visualizes **model misclassifications**, showing how often high and low hospitalization rates were correctly predicted.

![Flu ML Confusion Matrix](ml_prediction/flu_ml_confusion_matrix-age.png) ![Flu ML Confusion Matrix](ml_prediction/flu_ml_confusion_matrix-race.png)

---
