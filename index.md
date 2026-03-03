<!-- Count projects for context -->
{% assign total_projects = site.data.repos | size %}
<p class="lead mb-4">Currently showcasing <strong>{{ total_projects }}</strong> projects across different areas.</p>

<!-- Define the custom order for categories -->
{% assign category_order = "Personal / Profile, Data Analytics / SQL / Projects, Machine Learning / Deep Learning, Power BI / Visual Analytics, Python / Utilities / Practiques, Web / Portfolio / Pages" | split: ", " %}

<!-- Loop through each category in the defined order -->
{% for category in category_order %}
  {% assign category_projects = site.data.repos | where_exp: "repo", "repo.categories contains category" %}
  {% if category_projects.size > 0 %}
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