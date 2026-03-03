---
layout: page
title: "Data & ML Portfolio"
permalink: /
---
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">

<!-- Introduction with photo -->
<div style="display: flex; align-items: center; gap: 20px; margin-bottom: 20px;">
  <img src="/assets/images/meva-foto.png" alt="Oriac Gimeno" style="width: 150px; border-radius: 50%;">
  <div>
    <h1>👋 Hi, I'm <strong>Oriac Gimeno</strong></h1>
    <p>Welcome to my professional portfolio. Here you'll find a selection of projects I've developed in <strong>Data Science, Machine Learning, Deep Learning, and Visual Analytics</strong>. Each project may appear in multiple categories based on the technologies involved.</p>
  </div>
</div>

<!-- Count projects for context -->
{% assign total_projects = site.data.repos | size %}
<p class="lead mb-4">Currently showcasing <strong>{{ total_projects }}</strong> projects across different areas.</p>

<!-- Get unique categories from all repos -->
{% assign all_categories = site.data.repos | map: 'categories' | flatten | uniq | sort %}

<!-- Loop through each category -->
{% for category in all_categories %}
  {% assign category_projects = site.data.repos | where_exp: "repo", "repo.categories contains category" %}
  
  <div class="category-section mb-5">
    <h2 class="category-title">{{ category }}</h2>
    <p class="text-muted">{{ category_projects | size }} project(s)</p>

    <div class="row">
      {% for repo in category_projects %}
        <div class="col-md-6 col-lg-4 mb-4">
          <div class="card h-100 shadow-sm">
            <div class="card-body d-flex flex-column">
              <h5 class="card-title">
                <a href="{{ repo.url }}" target="_blank" rel="noopener noreferrer">{{ repo.name }}</a>
              </h5>
              
              {% if repo.desc %}
                <p class="card-text">{{ repo.desc }}</p>
              {% else %}
                <p class="card-text text-muted fst-italic">No description provided.</p>
              {% endif %}
              
              <!-- Optional: technology badges (activate when you add 'tech' field to repos.yml) -->
              <!-- 
              {% if repo.tech %}
                <div class="mb-2">
                  {% for tech in repo.tech %}
                    <span class="badge bg-secondary me-1">{{ tech }}</span>
                  {% endfor %}
                </div>
              {% endif %}
              -->
              
              <a href="{{ repo.url }}" class="btn btn-outline-primary btn-sm mt-auto align-self-start" target="_blank" rel="noopener noreferrer">
                <i class="fab fa-github me-1"></i> View on GitHub
              </a>
            </div>
          </div>
        </div>
      {% endfor %}
    </div>
  </div>
{% endfor %}

<!-- Footer with social links and icons -->
<hr class="mt-5">
<p class="text-center text-muted">
  📫 Connect with me:
  <a href="https://github.com/oriac-gimeno" target="_blank" rel="noopener noreferrer" class="mx-2">
    <i class="fab fa-github"></i> GitHub
  </a> •
  <a href="https://www.linkedin.com/in/oriacgimeno" target="_blank" rel="noopener noreferrer" class="mx-2">
    <i class="fab fa-linkedin"></i> LinkedIn
  </a>
</p>
