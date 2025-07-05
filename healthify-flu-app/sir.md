# SIR Model: Simulating Disease Spread

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

This interactive **SIR Model Simulation** allows you to adjust the **Basic Reproduction Number (R₀)** and observe the impact on the number of **Susceptible (S), Infected (I), and Recovered (R)** individuals over time.

### 🔹 How to Use:
- **Adjust R₀**: Higher values indicate more infectious diseases.
- **Modify Simulation Duration**: Observe outbreak trends over different time frames.
- **Analyze the Epidemic Curve**: See how flu hospitalization spreads over time.

_Model parameters are based on Bayesian estimates from Osthus et al. (2017) and foundational work from Cohen et al. (2016), Keeling et al. (2008), and Rothman et al. (2008)._

---

## **Run the SIR Model Simulation Below:**

<iframe title="SIR Model Shiny App" 
        src="https://healthifywebapp.shinyapps.io/app_sir/" 
        width="100%" height="650px" 
        frameborder="0" 
        style="border: 1px solid #ddd; border-radius: 8px;">
</iframe>
