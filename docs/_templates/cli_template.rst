.. -*- mode: rst -*-
  
*prose* features some command line tools to perform a variety of tasks:

.. list-table::

  {% for command, info in data.commands.items() %}
  * - :ref:`{{ command }}`
    - {{ info.help }}
  {% endfor %}

These commands can be called as sub-commands of the main ``prose`` command, with

.. code-block:: console

  prose COMMAND [OPTIONS] [ARGS]

{% for command, info in data.commands.items() %}

.. _{{ command }}:

{{ command }}
-----

{{ info.help }}

.. code-block:: console

  prose {{ command }} [OPTIONS] [ARGS]

.. list-table::
  :header-rows: 1

  * - argument
    - type
    - default
    - description
  {% for arg in info.arguments %}
  * {% if arg.short %}
    - ``{{ arg.short }}``{% if arg.long %}, ``{{ arg.long }}`` {% endif %} 
    {% else %}
    - ``{{ arg.name }}``
    {% endif %}
    {% if arg.type %} 
    - *{{ arg.type }}* 
    {% endif %}
    {% if arg.choices %}
    - {% for choice in arg.choices %} ``{{ choice }}`` {% endfor %}
    {% endif %}
    - ``{{ arg.default }}``
    - {{ arg.help }}
  {% endfor %}

{% endfor %}