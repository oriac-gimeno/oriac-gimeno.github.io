---
layout: home
title: "Oriac Gimeno — Data & ML Portfolio"
---

Benvingut al meu portfolio.  
A continuació trobaràs els projectes organitzats per categories.

{% assign categories = site.data.repos | map: 'categories' | flatten | uniq | sort %}

{% for category in categories %}
## {{ category }}

<div class="grid">
  {% for repo in site.data.repos %}
    {% if repo.categories contains category %}
      <div class="grid__item">
        <div class="archive__item">
          <div class="archive__item-body">
            <h3 class="archive__item-title">
              <a href="{{ repo.url }}" target="_blank">{{ repo.name }}</a>
            </h3>
            {% if repo.desc %}
              <p>{{ repo.desc }}</p>
            {% endif %}
          </div>
        </div>
      </div>
    {% endif %}
  {% endfor %}
</div>

{% endfor %}