def complete_html(html, interpolated_coordinates):
    header = (f"<li class='xr-section-item'><input id='section-xdas-header' class='xr-section-summary-in'"
              f"  type='checkbox' checked><label for='section-xdas-header'"
              f"  class='xr-section-summary'>Interpolated Coordinates: <span>({len(interpolated_coordinates)})</span></label>"
              f"  <div class='xr-section-inline-details'></div>"
              f"  <div class='xr-section-details'>"
              f"    <ul class='xr-var-list'>")
    main = []
    for dim, coordinate in interpolated_coordinates.items():
        main.append(f"<li class='xr-var-item'>"
                    f"  <div class='xr-var-name'><span class='xr-has-index'>{dim}</span></div>"
                    f"  <div class='xr-var-dims'>({dim})</div>"
                    f"  <div class='xr-var-dtype'>{coordinate.dtype}</div>"
                    f"  <div class='xr-var-preview xr-preview'>{coordinate}</div> <input id='attrs-xdas-{dim}' "
                    f"    class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-xdas-{dim}' "
                    f"    title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'> "
                    f"    <use xlink:href='#icon-file-text2'></use></svg> </label><input "
                    f"    id='data-xdas-{dim}' class='xr-var-data-in' type='checkbox'>"
                    f"    <label for='data-xdas-{dim}' title='Show/Hide data repr'>"
                    f"    <svg class='icon xr-icon-database'> <use xlink:href='#icon-database'></use> </svg></label>"
                    f"  <div class='xr-var-attrs'>"
                    f"      <dl class='xr-attrs'></dl>"
                    f"  </div>"
                    f"  <div class='xr-var-data'>"
                    f"      <pre>{coordinate.tie_values}</pre>"
                    f"  </div>"
                    f"</li>")
    main = "".join(main)
    footer = (f"    </ul>"
              f"  </div>"
              f"</li>"
              )
    return html[:-17] + header + main + footer + html[-17:]
