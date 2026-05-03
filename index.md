---
layout: default
title: "Oriac Gimeno Lozano | Business Strategy & Data Portfolio"
permalink: /
---

<div style="max-width: 1100px; margin: 0 auto; padding: 20px;">

  <div style="display: flex; align-items: center; gap: 30px; margin-bottom: 30px; flex-wrap: wrap;">
    <img src="/assets/images/meva-foto.png" alt="Oriac Gimeno Lozano (Oriac Gimeno)" style="width: 150px; border-radius: 50%; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
    <div style="flex: 1; min-width: 300px;">
      <h1>👋 Hi, I'm <strong>Oriac Gimeno Lozano</strong></h1>
      <p style="font-size: 1.1em; color: #555; margin-bottom: 10px;"><strong>Business Development Director | B2B Commercial Strategy · Data-Driven Growth</strong></p>
      <p>Welcome to my professional portfolio. Here you'll find a selection of projects I've developed in <strong>Data Science, Machine Learning, and Strategic Analytics</strong>. Also known professionally as <strong>Oriac Gimeno</strong>, I bridge the gap between high-level business strategy and data-driven execution.</p>
      <div style="margin-top: 15px;">
        <a href="https://oriac-gimeno.framer.website/" class="btn btn-primary btn-sm" target="_blank" style="background-color: #0055FF; border: none; padding: 8px 16px; font-weight: 600; border-radius: 6px;">🚀 Oriac Gimeno Consulting</a>
      </div>
    </div>
  </div>

  {% assign total_projects = site.data.repos | size %}
  <p class="lead mb-4" style="color: #666;">Currently showcasing <strong>{{ total_projects }}</strong> projects across different areas.</p>

  <hr style="margin-bottom: 40px; opacity: 0.1;">

  {% assign category_order = "Personal / Profile, Data Analytics / SQL / Projects, Machine Learning / Deep Learning, Power BI / Visual Analytics, Python / Utilities / Practiques, Web / Portfolio / Pages" | split: ", " %}

  {% for category in category_order %}
    {% assign category_projects = site.data.repos | where_exp: "repo", "repo.categories contains category" %}
    {% if category_projects.size > 0 %}
    <div class="category-section mb-5">
      <h2 class="category-title" style="font-weight: 700; color: #222;">{{ category | escape }}</h2>
      <p class="text-muted" style="margin-bottom: 20px;">{{ category_projects.size }} project(s)</p>

      <div class="row">
        {% for repo in category_projects %}
        <div class="col-md-6 col-lg-4 mb-4">
          <div class="card h-100 shadow-sm" style="border-radius: 10px; border: 1px solid #eee;">
            <div class="card-body d-flex flex-column" style="padding: 20px;">
              <h5 class="card-title">
                <a href="{{ repo.url }}" target="_blank" rel="noopener noreferrer" style="color: #0055FF; text-decoration: none; font-weight: 600;">{{ repo.name | escape }}</a>
              </h5>
              {% if repo.desc %}
              <p class="card-text" style="font-size: 0.95em; color: #555;">{{ repo.desc | escape }}</p>
              {% else %}
              <p class="card-text text-muted fst-italic">No description provided.</p>
              {% endif %}
              <a href="{{ repo.url }}" class="btn btn-outline-primary btn-sm mt-auto align-self-start" target="_blank" rel="noopener noreferrer" style="border-radius: 5px;">
                <i class="fab fa-github me-1"></i> View on GitHub
              </a>
            </div>
          </div>
        </div>
        {% endfor %}
      </div>
    </div>
    {% endif %}
  {% endfor %}

  <footer style="margin-top: 60px; border-top: 1px solid #eee; padding-top: 30px;">
    <p class="text-center text-muted">
      📫 Connect with me:
      <a href="https://github.com/oriac-gimeno" target="_blank" rel="noopener noreferrer" class="mx-2" style="text-decoration: none;"><i class="fab fa-github"></i> GitHub</a> •
      <a href="https://www.linkedin.com/in/oriacgimeno" target="_blank" rel="noopener noreferrer" class="mx-2" style="text-decoration: none;"><i class="fab fa-linkedin"></i> LinkedIn</a> •
      <a href="https://oriac-gimeno.framer.website/" target="_blank" rel="noopener noreferrer" class="mx-2" style="text-decoration: none;"><i class="fas fa-briefcase"></i> Consulting</a>
    </p>
    <p class="text-center text-muted" style="font-size: 0.85em; margin-top: 20px;">
      © 2026 Oriac Gimeno Lozano. Data Strategist & Business Development Leader in Barcelona.
    </p>
  </footer>

</div>
