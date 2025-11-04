// ABOUTME: D3.js visualization for Barber Motorsports Park track analysis
// ABOUTME: Renders track surface with boundaries and interactive zoom/pan

(async function() {
    // Load track boundaries
    const boundaries = await d3.json('data/processed/track_boundaries.json');

    console.log(`Loaded track boundaries:`);
    console.log(`  Inner: ${boundaries.inner.length} points`);
    console.log(`  Outer: ${boundaries.outer.length} points`);

    // Set up SVG
    const svg = d3.select('#track-svg');
    const width = window.innerWidth;
    const height = window.innerHeight;

    svg.attr('width', width)
       .attr('height', height);

    // Create main group for zoom/pan
    const g = svg.append('g');

    // Calculate bounds from all boundary points
    const allPoints = [...boundaries.inner, ...boundaries.outer];
    const xExtent = d3.extent(allPoints, d => d.x);
    const yExtent = d3.extent(allPoints, d => d.y);

    const trackWidth = xExtent[1] - xExtent[0];
    const trackHeight = yExtent[1] - yExtent[0];

    // Add padding (10%)
    const padding = 0.1;
    const xPadding = trackWidth * padding;
    const yPadding = trackHeight * padding;

    // Create scales
    const xScale = d3.scaleLinear()
        .domain([xExtent[0] - xPadding, xExtent[1] + xPadding])
        .range([0, width]);

    const yScale = d3.scaleLinear()
        .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
        .range([height, 0]); // Invert Y axis (SVG coordinates)

    // Create closed polygon path
    // Go around outer edge clockwise, then inner edge counter-clockwise
    const outerClosed = [...boundaries.outer, boundaries.outer[0]];
    const innerReversed = [...boundaries.inner].reverse();
    const innerClosed = [...innerReversed, innerReversed[0]];

    // Combine into single polygon path
    const trackPolygon = [...outerClosed, ...innerClosed];

    // Line generator
    const line = d3.line()
        .x(d => xScale(d.x))
        .y(d => yScale(d.y))
        .curve(d3.curveLinear);

    // Draw track surface as filled polygon
    g.append('path')
        .datum(trackPolygon)
        .attr('class', 'track-surface')
        .attr('d', line)
        .style('fill', '#2a2a2a')
        .style('stroke', '#444')
        .style('stroke-width', 1);

    // Add zoom behavior
    const zoom = d3.zoom()
        .scaleExtent([0.5, 10])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });

    svg.call(zoom);

    // Initial zoom to fit
    const scale = 0.9 * Math.min(
        width / (trackWidth * (1 + 2 * padding)),
        height / (trackHeight * (1 + 2 * padding))
    );

    const centerX = (xExtent[0] + xExtent[1]) / 2;
    const centerY = (yExtent[0] + yExtent[1]) / 2;

    const translateX = width / 2 - xScale(centerX) * scale;
    const translateY = height / 2 - yScale(centerY) * scale;

    svg.call(zoom.transform, d3.zoomIdentity
        .translate(translateX, translateY)
        .scale(scale));

    console.log('Track visualization ready');
})();
