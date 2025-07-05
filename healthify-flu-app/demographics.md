<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Healthify - Influenza Data Visualization</title>
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

        /* Caption Styling */
        .caption {
            font-size: 16px;
            font-weight: normal;
            color: #333;
            margin-top: 10px;
            text-align: left;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
            padding-bottom: 20px;
        }

        /* Section Headers */
        h2 {
            color: #0047ab;
            margin-top: 40px;
        }

    </style>
</head>
<body>

    <!-- Clickable Healthify Banner -->
    <a href="index.html" class="healthify-banner">Healthify</a>

    <div class="container">
        
        <!-- Age Group Incidence Plot -->
        <h2>Age Group Incidence</h2>
        <div class="iframe-container">
            <iframe title="Age Group Incidence" 
                    aria-label="Multiple Lines" 
                    id="datawrapper-chart-QWBVI" 
                    src="https://datawrapper.dwcdn.net/QWBVI/1/" 
                    scrolling="no" frameborder="0" 
                    style="width: 0; min-width: 100% !important; border: none;" 
                    height="408" data-external="1">
            </iframe>
        </div>
        <p class="caption">
            As widely cited in the literature, elderly populations are at greater risk of morbidity and mortality from seasonal influenza (Langer et al. 2023, Awaidy et al. 2024). 
            Data trends show an increase across age groups, with the highest rates exhibited among individuals aged 75-84 years (257.34 per 100,000).
        </p>

        <!-- Race/Ethnicity Incidence Plot -->
        <h2>Race/Ethnicity Incidence</h2>
        <div class="iframe-container">
            <iframe title="Race/Ethnicity Incidence" 
                    aria-label="Multiple Lines" 
                    id="datawrapper-chart-4E6Lq" 
                    src="https://datawrapper.dwcdn.net/4E6Lq/1/" 
                    scrolling="no" frameborder="0" 
                    style="width: 0; min-width: 100% !important; border: none;" 
                    height="425" data-external="1">
            </iframe>
        </div>
        <p class="caption">
            Structural inequities persist in reducing the likelihood that Black, American Indian/Alaska Native, Latina/o, Asian populations, and individuals of low socioeconomic status will receive the influenza vaccine (Stafford et al. 2024, Black 2022, O’Halloran et al. 2021). 
            This trend is reflected in our 2025 data, where the highest median incidence rates are observed among Black (99.7 per 100,000) and American Indian/Alaskan Native (65.9 per 100,000) populations.
        </p>

        <!-- Sex Incidence Plot -->
        <h2>Sex Incidence</h2>
        <div class="iframe-container">
            <iframe title="Sex Incidence" 
                    aria-label="Interactive line chart" 
                    id="datawrapper-chart-hm1js" 
                    src="https://datawrapper.dwcdn.net/hm1js/1/" 
                    scrolling="no" frameborder="0" 
                    style="width: 0; min-width: 100% !important; border: none;" 
                    height="383" data-external="1">
            </iframe>
        </div>
        <p class="caption">
            While men and women face similar risk factors for seasonal influenza, pregnancy can be an additional risk factor (Darling et al. 2022, Abdullahi et al. 2021). 
            Our data aligns with this trend, showing higher median incidence rates among females (70.9 per 100,000) compared to males (64.3 per 100,000).
        </p>

    </div>

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
