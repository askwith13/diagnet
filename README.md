# DiagNet AI Rerouting Component

## Overview
DiagNet’s rerouting AI module demonstrates how advanced algorithms can optimize specimen transport and diagnostic referrals in a healthcare network. This component integrates seamlessly with existing infrastructure (Nikshay, LIMS, SMS, WhatsApp) to minimize cognitive load for end users—no additional apps or dashboards are required.

---

## ✨ Key Features

- **Route Optimization with Dijkstra’s Algorithm**  
  Efficiently computes the shortest and most reliable path for specimen transport across the diagnostic network.

- **Adaptive Referral Policy via Q-Learning**  
  Learns from patient outcomes to recommend the most effective diagnostic facilities, continuously improving over time.

- **Transparent Recommendations with SHAP Explanations**  
  Each rerouting suggestion is accompanied by interpretable SHAP values, boosting user confidence in AI-driven decisions.

- **Seamless Delivery**  
  Recommendations are delivered through existing systems:
  - **Nikshay** and **LIMS** integration  
  - **SMS** and **WhatsApp** fallback notifications  
  - No new app installation or login required

---

## 🎯 Purpose
This module is a **demonstration of the inner workings** of DiagNet’s rerouting AI. It is not a standalone product interface, but rather a transparent showcase of:
- How specimens are rerouted optimally
- How facility recommendations adapt based on outcomes
- How interpretability is embedded into decision support

---

## 🧩 Architecture Highlights
- **Graph-based routing**: Dijkstra’s algorithm ensures shortest-path specimen transport.
- **Reinforcement learning**: Q-learning policy adapts facility choices dynamically.
- **Explainability layer**: SHAP explanations provide human-readable rationale.
- **Notification pipeline**: Recommendations flow through existing health IT systems.

---

🚀 Getting Started
1. Clone the repository:
   `bash
   git clone https://github.com/<your-org>/diagnet.git
   `
2. Navigate to the rerouting component:
   `bash
   cd diagnet
   `
3. Review the sample workflows and code snippets to understand how the AI module operates.

---

📌 Notes
- This repository does not include a user-facing dashboard.  
- All recommendations are delivered via existing infrastructure to keep additional cognitive load minimal.  
- The focus is on algorithmic transparency and demonstration, not deployment-ready interfaces.

---

📜 License
This project is licensed under GPL v3.0 to ensure openness and collaborative improvement.

---

🤝 Contributing
Contributions are welcome! Please open issues or submit pull requests to enhance the demonstration, improve documentation, or extend functionality.

---

📧 Contact
For questions or collaboration inquiries, please reach out via the repository’s issue tracker or project maintainers.
`

---
