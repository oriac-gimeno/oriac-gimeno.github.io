---
layout: default
title: "Oriac Gimeno Lozano | Business Strategy & Data Portfolio"
permalink: /
---

<!-- Contenedor Principal para limitar el ancho y mejorar la estética -->
<div style="max-width: 1000px; margin: 0 auto; padding: 0 20px;">

  <!-- Introducción con foto y SEO optimizado -->
  <div style="display: flex; align-items: center; gap: 30px; margin-bottom: 40px; flex-wrap: wrap;">
    <img src="/assets/images/meva-foto.png" alt="Oriac Gimeno Lozano (Oriac Gimeno)" style="width: 150px; height: 150px; border-radius: 50%; object-fit: cover; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
    <div style="flex: 1; min-width: 300px;">
      <h1 style="margin-bottom: 10px;">👋 Hi, I'm <strong>Oriac Gimeno Lozano</strong></h1>
      <p style="font-size: 1.2em; color: #333; margin-bottom: 15px; line-height: 1.3;">
        <strong>Business Development Director | B2B Commercial Strategy · Data-Driven Growth</strong>
      </p>
      <p style="line-height: 1.6; color: #555;">
        Welcome to my professional portfolio. Here you'll find a selection of projects developed in <strong>Data Science, Machine Learning, and Strategic Analytics</strong>. Also known professionally as <strong>Oriac Gimeno</strong>, I bridge the gap between high-level business strategy and data-driven execution.
      </p>
      <div style="margin-top: 20px;">
        <a href="https://oriac-gimeno.framer.website/" class="btn btn-primary" target="_blank" style="background-color: #0055FF; border: none; padding: 10px 20px; font-weight: 600; border-radius: 8px;">🚀 Oriac Gimeno Consulting</a>
      </div>
    </div>
  </div>

  <!-- Contador total de proyectos -->
  {% assign total_projects = site.data.repos | size %}
  <p class="lead mb-5" style="border-left: 4px solid #0055FF; padding-left: 15px; font-style: italic;">
    Currently showcasing <strong>{{ total_projects }}</strong> projects across different areas.
  </p>

  <!-- ORDEN MANUAL DE CATEGORÍAS -->
  {% assign category_order = "Personal / Profile, Data Analytics / SQL / Projects, Machine Learning / Deep Learning, Power BI / Visual Analytics, Python / Utilities / Practiques, Web / Portfolio / Pages" | split: ", " %}

  <!-- BUCLE PARA CADA CATEGORÍA -->
  {% for category in category_order %}
    {% assign category_projects = site.data.repos | where_exp: "repo", "repo.categories contains category" %}
    {% if category_projects.size > 0 %}
  <div class="category-section mb-5">
    <h2 class="category-title" style="font-weight: 700; border-bottom: 2px solid #eee; padding-bottom: 10px;">{{ category | escape }}</h2>
    <p class="text-muted" style="margin-bottom: 25px;">{{ category_projects.size }} project(s)</p>

    <div class="row">
      {% for repo in category_projects %}
      <div class="col-md-6 mb-4"> <!-- Cambiado a col-md-6 para que las tarjetas respiren más -->
        <div class="card h-100 shadow-sm" style="border-radius: 12px; transition: transform 0.2s; border: 1px solid #eee;">
          <div class="card-body d-flex flex-column" style="padding: 25px;">
            <h5 class="card-title" style="margin-bottom: 15px;">
              <a href="{{ repo.url }}" target="_blank" rel="noopener noreferrer" style="color: #0055FF; text-decoration: none; font-weight: 600;">{{ repo.name | escape }}</a>
            </h5>
            {% if repo.desc %}
            <p class="card-text" style="color: #666; font-size: 0.95em; line-height: 1.5;">{{ repo.desc | escape }}</p>
            {% else %}
            <p class="card-text text-muted fst-italic">No description provided.</p>
            {% endif %}
            <div class="mt-auto pt-3">
              <a href="{{ repo.url }}" class="btn btn-outline-dark btn-sm" target="_blank" rel="noopener noreferrer" style="border-radius: 6px;">
                <i class="fab fa-github me-1"></i> View on GitHub
              </a>
            </div>
          </div>
        </div>
      </div>
      {% endfor %}
    </div>
  </div>
    {% endif %}
  {% endfor %}

  <!-- Footer con enlaces sociales -->
  <footer style="margin-top: 80px; padding-bottom: 40px; border-top: 1px solid #eee; pt-4">
    <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 20px; flex-wrap: wrap;">
      <a href="https://github.com/oriac-gimeno" target="_blank" rel="noopener noreferrer" style="text-decoration: none; color: #333;"><i class="fab fa-github"></i> GitHub</a>
      <a href="https://www.linkedin.com/in/oriacgimeno" target="_blank" rel="noopener noreferrer" style="text-decoration: none; color: #0077B5;"><i class="fab fa-linkedin"></i> LinkedIn</a>
      <a href="https://oriac-gimeno.framer.website/" target="_blank" rel="noopener noreferrer" style="text-decoration: none; color: #0055FF;"><i class="fas fa-briefcase"></i> Consulting</a>
    </div>
    <p class="text-center text-muted" style="font-size: 0.85em;">
      © 2026 Oriac Gimeno Lozano. Data Strategist & Business Development Leader. Barcelona.
    </p>
  </footer>

</div> <!-- Cierre del contenedor principal -->
