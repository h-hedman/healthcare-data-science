<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Flu Symptom NLP Analysis</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 0;
      background-color: #f4f4f4;
      text-align: center;
    }

    #chartContainer {
      width: 80%;
      max-width: 600px;
      margin: 20px auto;
      background-color: #f4f4f4;
    }

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
      position: fixed;
      top: 0;
      left: 0;
      z-index: 1000;
    }

    .healthify-banner:hover {
      background-color: #003580;
      transform: scale(1.03);
    }

    .container {
      max-width: 800px;
      margin: 80px auto 40px auto;
      padding: 20px;
    }

    .iframe-container {
      width: 100%;
      max-width: 1200px;
      margin: auto;
      overflow: hidden;
      position: relative;
      padding-top: 56.25%;
    }

    .iframe-container iframe {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      border: none;
      border-radius: 8px;
    }

    iframe + div {
      display: none !important;
    }
  </style>
</head>
<body>

  <a href="index.html" class="healthify-banner">Healthify</a>

  <div class="iframe-container">
    <iframe 
      title="Flu NLP Symptoms Shiny App" 
      src="https://healthifywebapp.shinyapps.io/app_nlp_symptoms/" 
      allowfullscreen>
    </iframe>
  </div>

  <br><br>
  <div><h3>Automatic Clinical Notes Analysis Based on Sample CDC Data</h3></div>
  <br>
  <div id="rowCount">Loading...</div>
  <div id="chartContainer">
    <canvas id="myChart"></canvas>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script>
    fetch('test_data.txt')
      .then(response => {
        if (!response.ok) throw new Error('Network response was not ok');
        return response.text();
      })
      .then(data => {
        const rows = data.split('\n').map(line => line.trim()).filter(line => line !== '');
        document.getElementById('rowCount').textContent = "Number of rows: " + rows.length;

        const ageFrequency = {};
        const ageRegex = /Patient Age:\s*(\d+)/;

        rows.forEach(row => {
          const match = row.match(ageRegex);
          if (match && match[1]) {
            const age = match[1];
            ageFrequency[age] = (ageFrequency[age] || 0) + 1;
          }
        });

        const labels = Object.keys(ageFrequency)
          .map(Number).sort((a, b) => a - b).map(String);
        const counts = labels.map(age => ageFrequency[age]);

        const ctx = document.getElementById('myChart').getContext('2d');
        new Chart(ctx, {
          type: 'bar',
          data: {
            labels: labels,
            datasets: [{
              label: 'Frequency',
              data: counts,
              backgroundColor: 'rgba(75, 192, 192, 0.2)',
              borderColor: 'rgba(75, 192, 192, 1)',
              borderWidth: 1
            }]
          },
          options: {
            scales: {
              y: {
                beginAtZero: true,
                ticks: {
                  precision: 0
                }
              }
            }
          }
        });
      })
      .catch(error => {
        console.error('Error while fetching the text file:', error);
        document.getElementById('rowCount').textContent = "Error loading file.";
      });
  </script>

</body>
</html>
