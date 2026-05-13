---
layout: default
title: "Oriac Gimeno Lozano | Business Strategy & Data Portfolio"
permalink: /
---

<div class="mb-5">
  <p class="display-6" style="font-weight: 300; color: #333;">
    Bridging the gap between <strong>high-level business strategy</strong> and <strong>data-driven execution</strong>.
  </p>
  <p class="lead text-secondary">
    Welcome to my professional portfolio. Here you'll find a curated selection of projects in <strong>Data Science, Machine Learning, and Strategic Analytics</strong> developed to solve real-world commercial challenges.
  </p>
  
  <div class="mt-4 d-flex align-items-center gap-3">
    <a href="https://oriac-gimeno.framer.website/" class="btn btn-lg text-white shadow-sm" target="_blank" style="background-color: #0055FF; border-radius: 8px; padding: 12px 25px; font-weight: 600;">
      🚀 Access Consulting Site
    </a>
    <span class="text-muted small">| Professional advisory for GTM Strategy</span>
  </div>
</div>

<hr class="my-5" style="opacity: 0.1;">

{% assign total_projects = site.data.repos | size %}
<div class="d-flex justify-content-between align-items-end mb-4">
  <h3 style="font-weight: 700; margin-bottom: 0;">Technical Portfolio</h3>
  <span class="badge rounded-pill bg-light text-dark border" style="font-size: 0.9rem; padding: 8px 15px;">
    {{ total_projects }} Projects
  </span>
</div>

{% assign category_order = "Personal / Profile, Data Analytics / SQL / Projects, Machine Learning / Deep Learning, Power BI / Visual Analytics, Python / Utilities / Practiques, Web / Portfolio / Pages" | split: ", " %}

{% for category in category_order %}
  {% assign category_projects = site.data.repos | where_exp: "repo", "repo.categories contains category" %}
  {% if category_projects.size > 0 %}
<section class="mb-5">
  <div class="d-flex align-items-center mb-3">
    <div style="height: 2px; width: 30px; background-color: #0055FF; margin-right: 15px;"></div>
    <h4 class="text-uppercase tracking-wider" style="font-size: 0.9rem; font-weight: 800; letter-spacing: 1px; margin-bottom: 0;">
      {{ category | escape }}
    </h4>
  </div>

  <div class="row g-4">
    {% for repo in category_projects %}
    <div class="col-md-6 col-lg-4">
      <div class="card h-100 shadow-sm border-0" style="transition: transform 0.2s; border-radius: 12px;">
        <div class="card-body p-4 d-flex flex-column">
          <h5 class="card-title mb-3">
            <a href="{{ repo.url }}" target="_blank" class="text-decoration-none" style="color: #0055FF; font-weight: 700;">
              {{ repo.name | escape }}
            </a>
          </h5>
          <p class="card-text text-muted small mb-4" style="line-height: 1.6;">
            {% if repo.desc %}{{ repo.desc | escape }}{% else %}Strategic data implementation project.{% endif %}
          </p>
          <a href="{{ repo.url }}" class="btn btn-link p-0 mt-auto text-dark fw-bold text-decoration-none small" target="_blank">
            <i class="fab fa-github me-1"></i> Documentation →
          </a>
        </div>
      </div>
    </div>
    {% endfor %}
  </div>
</section>
  {% endif %}
{% endfor %}

<footer class="mt-5 pt-5 border-top">
  <div class="row align-items-center">
    <div class="col-md-6 text-center text-md-start">
      <p class="text-muted small mb-0">© 2026 Oriac Gimeno Lozano | Barcelona</p>
    </div>
    <div class="col-md-6 text-center text-md-end">
      <a href="https://github.com/oriac-gimeno" target="_blank" class="text-dark mx-2"><i class="fab fa-github"></i></a>
      <a href="https://www.linkedin.com/in/oriacgimeno" target="_blank" class="text-dark mx-2"><i class="fab fa-linkedin"></i></a>
    </div>
  </div>
</footer>
