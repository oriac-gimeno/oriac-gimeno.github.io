---
layout: default
title: "Oriac Gimeno Lozano | Business Strategy & Data Portfolio"
permalink: /
---

<div class="mb-4">
  <p class="lead">Welcome to my professional portfolio. Here you'll find a selection of projects I've developed in <strong>Data Science, Machine Learning, and Strategic Analytics</strong>.</p>
  <p>Also known professionally as <strong>Oriac Gimeno</strong>, I bridge the gap between high-level business strategy and data-driven execution.</p>
  <div class="mt-3">
    <a href="https://oriac-gimeno.framer.website/" class="btn btn-primary" target="_blank" style="background-color: #0055FF; border: none; padding: 10px 20px;">
      🚀 Oriac Gimeno Consulting
    </a>
  </div>
</div>

{% assign total_projects = site.data.repos | size %}
<p class="mb-4 text-muted">Currently showcasing <strong>{{ total_projects }}</strong> projects across different areas.</p>

{% assign category_order = "Personal / Profile, Data Analytics / SQL / Projects, Machine Learning / Deep Learning, Power BI / Visual Analytics, Python / Utilities / Practiques, Web / Portfolio / Pages" | split: ", " %}

{% for category in category_order %}
  {% assign category_projects = site.data.repos | where_exp: "repo", "repo.categories contains category" %}
  {% if category_projects.size > 0 %}
<div class="category-section mb-5">
  <h2 class="h3 border-bottom pb-2">{{ category | escape }}</h2>
  <p class="text-muted" style="font-size: 0.9rem;">{{ category_projects.size }} project(s)</p>

  <div class="row">
    {% for repo in category_projects %}
    <div class="col-md-6 col-lg-4 mb-4">
      <div class="card h-100 shadow-sm border-0" style="background: white;">
        <div class="card-body d-flex flex-column">
          <h5 class="card-title">
            <a href="{{ repo.url }}" target="_blank" rel="noopener noreferrer" class="text-decoration-none" style="color: #0055FF;">{{ repo.name | escape }}</a>
          </h5>
          {% if repo.desc %}
          <p class="card-text text-secondary small">{{ repo.desc | escape }}</p>
          {% else %}
          <p class="card-text text-muted fst-italic small">No description provided.</p>
          {% endif %}
          <a href="{{ repo.url }}" class="btn btn-outline-dark btn-sm mt-auto align-self-start" target="_blank" rel="noopener noreferrer">
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

<hr class="mt-5">
<p class="text-center text-muted">
  📫 Connect with me:
  <a href="https://github.com/oriac-gimeno" target="_blank" rel="noopener noreferrer" class="mx-2 text-decoration-none"><i class="fab fa-github"></i> GitHub</a> •
  <a href="https://www.linkedin.com/in/oriacgimeno" target="_blank" rel="noopener noreferrer" class="mx-2 text-decoration-none"><i class="fab fa-linkedin"></i> LinkedIn</a> •
  <a href="https://oriac-gimeno.framer.website/" target="_blank" rel="noopener noreferrer" class="mx-2 text-decoration-none"><i class="fas fa-briefcase"></i> Consulting</a>
</p>
<p class="text-center text-muted" style="font-size: 0.8em; margin-top: 20px;">
  © 2026 Oriac Gimeno Lozano. Data Strategist & Business Development Leader in Barcelona.
</p>
