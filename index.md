---

layout: default

title: "Oriac Gimeno Lozano | Business Strategy & Data Portfolio"

permalink: /

---



<!-- Introducción con foto y SEO optimizado -->

<div style="display: flex; align-items: center; gap: 20px; margin-bottom: 20px;">

  <img src="/assets/images/meva-foto.png" alt="Oriac Gimeno Lozano (Oriac Gimeno)" style="width: 150px; border-radius: 50%;">

  <div>

    <h1>👋 Hi, I'm <strong>Oriac Gimeno Lozano</strong></h1>

    <p style="font-size: 1.1em; color: #555; margin-bottom: 10px;"><strong>Business Development Director | B2B Commercial Strategy · Data-Driven Growth</strong></p>

    <p>Welcome to my professional portfolio. Here you'll find a selection of projects I've developed in <strong>Data Science, Machine Learning, and Strategic Analytics</strong>. Also known professionally as <strong>Oriac Gimeno</strong>, I bridge the gap between high-level business strategy and data-driven execution.</p>

    <div style="margin-top: 15px;">

      <a href="https://oriac-gimeno.framer.website/" class="btn btn-primary btn-sm" target="_blank" style="background-color: #0055FF; border: none;">🚀 Oriac Gimeno Consulting</a>

    </div>

  </div>

</div>



<!-- Contador total de proyectos -->

{% assign total_projects = site.data.repos | size %}

<p class="lead mb-4">Currently showcasing <strong>{{ total_projects }}</strong> projects across different areas.</p>



<!-- ORDEN MANUAL DE CATEGORÍAS (UNA SOLA LÍNEA) -->

{% assign category_order = "Personal / Profile, Data Analytics / SQL / Projects, Machine Learning / Deep Learning, Power BI / Visual Analytics, Python / Utilities / Practiques, Web / Portfolio / Pages" | split: ", " %}



<!-- BUCLE PARA CADA CATEGORÍA -->

{% for category in category_order %}

  {% assign category_projects = site.data.repos | where_exp: "repo", "repo.categories contains category" %}

  {% if category_projects.size > 0 %}

<div class="category-section mb-5">

  <h2 class="category-title">{{ category | escape }}</h2>

  <p class="text-muted">{{ category_projects.size }} project(s)</p>



  <div class="row">

    {% for repo in category_projects %}

    <div class="col-md-6 col-lg-4 mb-4">

      <div class="card h-100 shadow-sm">

        <div class="card-body d-flex flex-column">

          <h5 class="card-title">

            <a href="{{ repo.url }}" target="_blank" rel="noopener noreferrer">{{ repo.name | escape }}</a>

          </h5>

          {% if repo.desc %}

          <p class="card-text">{{ repo.desc | escape }}</p>

          {% else %}

          <p class="card-text text-muted fst-italic">No description provided.</p>

          {% endif %}

          <a href="{{ repo.url }}" class="btn btn-outline-primary btn-sm mt-auto align-self-start" target="_blank" rel="noopener noreferrer">

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



<!-- Footer con enlaces sociales y texto defensivo SEO -->

<hr class="mt-5">

<p class="text-center text-muted">

  📫 Connect with me:

  <a href="https://github.com/oriac-gimeno" target="_blank" rel="noopener noreferrer" class="mx-2"><i class="fab fa-github"></i> GitHub</a> •

  <a href="https://www.linkedin.com/in/oriacgimeno" target="_blank" rel="noopener noreferrer" class="mx-2"><i class="fab fa-linkedin"></i> LinkedIn</a> •

  <a href="https://oriac-gimeno.framer.website/" target="_blank" rel="noopener noreferrer" class="mx-2"><i class="fas fa-briefcase"></i> Consulting</a>

</p>

<p class="text-center text-muted" style="font-size: 0.8em; margin-top: 20px;">

  © 2026 Oriac Gimeno Lozano. Data Strategist & Business Development Leader in Barcelona.

</p>


