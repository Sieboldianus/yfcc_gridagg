{% extends 'full.tpl'%}

{# this template will render cells with tags highlight,
highlight_red and hide_code differently #}

{% block input_group %}
    {% if 'highlight_red' in cell['metadata'].get('tags', []) %}
        <div style="background-color:#FFF0F2">
            {{ super() }}
        </div>
    {% elif 'highlight' in cell['metadata'].get('tags', []) %}
        <div style="background-color:#E0F0F5">
            {{ super() }}
        </div>
    {% else %}
        {% if 'hide_code' in cell['metadata'].get('tags', []) %}
            <div style="padding-left: 40px; font-size: 20px;">•••</div>
        {% else %}
            {{ super() }}
        {% endif %}
    {% endif %}
{% endblock input_group %}

{% block output_group %}
    {{ super() }}
{% endblock output_group %}
