<style>
/* Healthify Button Styling */
.healthify-button {
    position: absolute;
    top: 10px;
    left: 10px;
    padding: 10px 20px;
    font-size: 18px;
    font-weight: bold;
    color: white;
    background-color: #0073e6;
    border-radius: 8px;
    text-decoration: none;
    transition: background 0.3s;
}
.healthify-button:hover {
    background-color: #005bb5;
}
iframe + div {
    display: none !important;
}
</style>

<!-- Healthify Home Button (Top Left) -->
<a href="https://github.gatech.edu/pages/hhedman3/cs-6440-practicum-healthify/index.html" class="healthify-button">Healthify</a>

---

<!-- Datawrapper Chart Embed -->
<iframe title="Flu Hospitalizations Column Chart" 
        aria-label="Column Chart" 
        id="datawrapper-chart-5YZlv" 
        src="https://datawrapper.dwcdn.net/5YZlv/1/" 
        scrolling="no" frameborder="0" 
        style="width: 0; min-width: 100% !important; border: none;" 
        height="390" data-external="1">
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
