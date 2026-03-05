/** @type {import('next').NextConfig} */
const nextConfig = {
    output: "export", // Enables static HTML export for GitHub Pages
    images: {
        unoptimized: true, // Required for static exports if you add images later
    },
    // If your repo is named "ClinicalVisionDFU" and not a custom domain:
    basePath: "/ClinicalVisionDFU",
};
export default nextConfig;