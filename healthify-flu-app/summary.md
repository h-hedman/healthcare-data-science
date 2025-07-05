<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flu Data Summary</title>
    <style>
        /* Reset & Global Styles */
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
            text-align: center;
        }

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

        /* Content Container */
        .container {
            max-width: 800px;
            margin: 80px auto 40px auto; /* Adjusted to avoid overlap with fixed banner */
            padding: 20px;
        }

        /* Responsive iframe container */
        .iframe-container {
            width: 100%;
            max-width: 1200px;
            margin: auto;
            overflow: hidden;
            position: relative;
            padding-top: 56.25%; /* Maintain aspect ratio */
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

        /* Hide unnecessary iframe overlays */
        iframe + div {
            display: none !important;
        }

    </style>
</head>
<body>

    <!-- Clickable Healthify Banner (Returns to Homepage) -->
    <a href="index.html" class="healthify-banner">Healthify</a>

    <!-- Content Section -->
    <div class="container">
        <h2>Median Confirmed Hospitalizations of Seasonal Flu Incidence per 100,000 Population</h2>

        <!-- New Datawrapper Chart Embed (Flu Incidence Trends) -->
        <div class="iframe-container">
            <iframe title="Flu Incidence Trends" 
                    aria-label="Interactive line chart" 
                    id="datawrapper-chart-jPRTH" 
                    src="https://datawrapper.dwcdn.net/jPRTH/1/" 
                    scrolling="no" frameborder="0" 
                    style="width: 0; min-width: 100% !important; border: none;" 
                    height="383" data-external="1">
            </iframe>
        </div>

        <!-- Source Notes -->
        <p style="font-size: 14px; color: #555;">
            Influenza Hospitalization Surveillance Network (FluSurv-NET) conducts population-based surveillance for laboratory-confirmed influenza-associated hospitalizations in children (persons younger than 18 years) and adults.
            <br>
            <a href="https://gis.cdc.gov/GRASP/Fluview/" target="_blank">Source: CDC FluSurv-NET</a>
        </p>
    </div>

    <!-- Script to Adjust Datawrapper Heights -->
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

</body>
</html>
