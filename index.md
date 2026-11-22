---
layout: default
title: "Oriac Gimeno Lozano | Business Strategy & Data Portfolio"
permalink: /
---

<div class="mb-5">
  <!-- The page had no h1 at all, which Bing's crawler reported as an error alongside
       the missing title and description. The name goes in the h1 because that is the
       strongest text signal on the page and the page is this person's portfolio; it is
       sized small on purpose, since the weight of an h1 is structural, not visual. -->
  <h1 class="h4 mb-3" style="font-weight: 600; color: #212529;">Oriac Gimeno Lozano</h1>
  <p class="display-6" style="font-weight: 300; color: #333;">
    Bridging the gap between <strong>high-level business strategy</strong> and <strong>data-driven execution</strong>.
  </p>
  <p class="lead text-secondary">
    Welcome to my professional portfolio. Here you'll find a curated selection of projects in <strong>Data Science, Machine Learning, and Strategic Analytics</strong> developed to solve real-world commercial challenges.
  </p>
  
  <style>
    .brand-row { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; max-width: 640px; }
    .brand-btn { display: inline-flex; align-items: center; justify-content: center; gap: 10px; min-height: 48px; padding: 10px 18px; border: 1px solid #d0d5dd; border-radius: 999px; background: #fff; color: #212529; font-weight: 600; text-decoration: none; white-space: nowrap; box-shadow: 0 1px 2px rgba(0,0,0,.05); transition: border-color .15s, box-shadow .15s, transform .15s; }
    .brand-btn:hover, .brand-btn:focus-visible { border-color: #0055FF; color: #212529; box-shadow: 0 4px 12px rgba(0,85,255,.15); transform: translateY(-1px); }
    .brand-logo { width: 20px; height: 20px; flex: none; object-fit: contain; border-radius: 4px; }
    .brand-btn .fa-linkedin { font-size: 20px; line-height: 20px; width: 20px; text-align: center; }
    @media (max-width: 575.98px) { .brand-row { grid-template-columns: 1fr; max-width: none; } }
  </style>
  <div class="mt-4 brand-row">
    <a href="https://oriac-gimeno.framer.website/" class="brand-btn" target="_blank" rel="noopener">
      <img class="brand-logo" src="{{ site.baseurl }}/assets/images/og-monogram.png" alt="" width="20" height="20"> OG Consulting
    </a>
    <a href="https://oriacgimeno.com/" class="brand-btn" target="_blank" rel="noopener">
      <svg class="brand-logo" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="6" fill="#faf7f2"/><path d="M16.07 6.58Q18.39 6.58 20.15 7.52Q21.90 8.46 22.88 10.15Q23.87 11.84 23.87 14.07Q23.87 16.19 22.86 17.87Q21.84 19.55 20.06 20.52Q18.27 21.50 15.92 21.50Q13.61 21.50 11.86 20.54Q10.11 19.58 9.12 17.89Q8.13 16.20 8.13 14Q8.13 11.86 9.15 10.18Q10.16 8.51 11.94 7.54Q13.73 6.58 16.07 6.58M17.13 19.93Q18.15 19.77 18.77 18.99Q19.40 18.20 19.57 16.82Q19.74 15.44 19.38 13.50Q19.04 11.57 18.38 10.32Q17.72 9.07 16.82 8.51Q15.92 7.96 14.87 8.13Q13.83 8.30 13.21 9.09Q12.59 9.88 12.43 11.25Q12.27 12.62 12.62 14.57Q12.96 16.49 13.62 17.74Q14.28 19 15.18 19.55Q16.08 20.11 17.13 19.93" fill="#1a1814"/><rect x="9" y="25" width="14" height="2" fill="#1e3a5f"/></svg> oriacgimeno.com
    </a>
    <a href="https://www.linkedin.com/in/oriacgimeno" class="brand-btn" target="_blank" rel="noopener">
      <i class="fab fa-linkedin" aria-hidden="true"></i> LinkedIn
    </a>
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
      <a href="https://oriacgimeno.com/" target="_blank" class="text-dark mx-2 d-inline-flex align-items-center justify-content-center" style="min-height: 44px; min-width: 44px; font-size: 18px; gap: 8px;"><i class="fas fa-globe" style="font-size: 26px;"></i> oriacgimeno.com</a>
      <a href="https://github.com/oriac-gimeno" target="_blank" class="text-dark mx-2 d-inline-flex align-items-center justify-content-center" style="min-height: 44px; min-width: 44px;" aria-label="GitHub"><i class="fab fa-github" style="font-size: 26px;"></i></a>
      <a href="https://www.linkedin.com/in/oriacgimeno" target="_blank" class="text-dark mx-2 d-inline-flex align-items-center justify-content-center" style="min-height: 44px; min-width: 44px;" aria-label="LinkedIn"><i class="fab fa-linkedin" style="font-size: 26px;"></i></a>
    </div>
  </div>
</footer>
