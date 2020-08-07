import React from 'react';

import './Background.css';

function Background() {
  const [image, setImage] = React.useState(null);

  React.useEffect(() => {
    fetch('https://source.unsplash.com/1920x1080/?Wallpaper')
      .then((response) => {
        if (!response.ok) {
          throw Error('Error fetching image!');
        }
        return response.url;
      })
      .then((data) => setImage(data as any))
      .catch((err) => {
        throw Error(err.message);
      });
  }, []);

  if (image == null) {
    return <div></div>;
  } else {
    console.log(image);
    return (
      <div style={{ height: '100%' }}>
        <div
          className='background-image'
          style={{ backgroundImage: `url(${image})` }}
        />
      </div>
    );
  }
}

export { Background };
