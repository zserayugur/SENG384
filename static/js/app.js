function formatNumber(value, digits = 3) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return value;
    }

    if (Math.abs(number) >= 1000000) {
        return number.toExponential(3);
    }

    return number.toFixed(digits);
}

const analyzeBtn = document.getElementById("analyzeBtn");

if (analyzeBtn) {
    analyzeBtn.addEventListener("click", async () => {
        const original = document.getElementById("original").value;
        const transformed = document.getElementById("transformed").value;

        try {
            const response = await fetch("/analyze/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    original_path: original,
                    transformed_path: transformed
                })
            });

            const data = await response.json();

            if (!data.success) {
                document.getElementById("metricsResult").innerHTML =
                    `<p style="color:red;">${data.message}</p>`;
                return;
            }

            const result = data.data;

            document.getElementById("metricsResult").innerHTML = `
                <p><strong>MSE:</strong> ${formatNumber(result.metrics.mse, 2)}</p>
                <p><strong>PSNR:</strong> ${formatNumber(result.metrics.psnr, 2)} dB</p>
                <p><strong>SSIM:</strong> ${formatNumber(result.metrics.ssim, 3)}</p>
            `;

            document.getElementById("energyResult").innerHTML = `
                <p><strong>Original Energy:</strong> ${formatNumber(result.energy.original, 3)}</p>
                <p><strong>Transformed Energy:</strong> ${formatNumber(result.energy.transformed, 3)}</p>
                <p><strong>Original Ratio:</strong> ${formatNumber(result.energy.original_ratio, 5)}</p>
                <p><strong>Transformed Ratio:</strong> ${formatNumber(result.energy.transformed_ratio, 5)}</p>
            `;

            document.getElementById("originalSpectrum").src =
                "/static/results/original_spectrum.png?t=" + new Date().getTime();

            document.getElementById("transformedSpectrum").src =
                "/static/results/transformed_spectrum.png?t=" + new Date().getTime();

        } catch (error) {
            document.getElementById("metricsResult").innerHTML =
                `<p style="color:red;">Request failed: ${error}</p>`;
        }
    });
}